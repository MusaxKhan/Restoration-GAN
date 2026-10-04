"""Record the demo video of the running app (Docker stack on :3000, optional MLflow on :5000).

Drives the real UI with Playwright (screen recording of the browser), narrates with the Windows text-to-speech voice and
muxes everything with ffmpeg:
    PLAYWRIGHT_BROWSERS_PATH=E:/tools/pw python scripts/record_demo.py --out demo/demo.mp4
Needs: Chrome, ffmpeg, Windows PowerShell (System.Speech), docker stack running, mlflow service running.
"""
import argparse
import html
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = shutil.which("ffmpeg") or r"C:\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
DOCKER = r"E:\Docker\resources\bin\docker.exe"

NARRATION = {
    "intro": "Hello. This is the demo of my Generative A I assignment one: image restoration with autoencoders, a "
             "mixture of experts, and a style conditioned face to sketch generator. Everything runs as one Docker "
             "Compose stack, a FastAPI backend serving seven ONNX models and a React frontend that I first designed "
             "in Google Stitch.",
    "pipeline": "The project has four tasks on one web application. Task one is a universal denoising autoencoder "
                "trained with an L one plus S S I M loss. Task two adds a classifier that recognises the corruption "
                "and routes the image to one of three specialist autoencoders. Task three replaces the hard decision "
                "with a soft mixture of experts, whose gate is initialised from the classifier and fine tuned jointly "
                "with a balance loss. Task four is a style conditioned pix two pix generator that turns a face "
                "photograph into a pencil sketch. Every task was tuned with Optuna and tracked in M L Flow.",
    "docker": "The whole system starts with one command, docker compose up dash dash build. Here are the running "
              "containers: the backend with all seven models loaded, and the nginx frontend on port three thousand. "
              "The models themselves are downloaded from a GitHub release by a small script.",
    "results": "These are the measured test results. The classifier reaches ninety nine point six percent accuracy. "
               "The universal autoencoder improves S S I M from zero point six seven one to zero point seven two "
               "three, although its P S N R is slightly lower than the noisy input, which the report discusses "
               "openly. Hard routing gives twenty four point seven nine decibels, with oracle and predicted routing "
               "identical, and the soft mixture of experts is best at twenty six point two eight decibels.",
    "universal": "Task one, the universal denoising autoencoder. I pick a sample image, add salt and pepper noise at "
                 "high strength, and press restore. The app shows the corrupted input, the restored output, an error "
                 "map, and the P S N R and S S I M against the clean image, plus the inference time.",
    "universal2": "Now the same single model on Gaussian blur at medium strength, with a different image. One model "
                  "handles every corruption type, which is why its improvement is modest. The specialists in the "
                  "next tasks do better.",
    "hard": "Task two, hard routing. A classifier first predicts the corruption type, here rectangular occlusion, and "
            "shows its probabilities. The image is then sent to the matching specialist autoencoder. The panel shows "
            "which expert was selected, and the classifier and expert times separately.",
    "hard2": "A different image with salt and pepper noise. The classifier now selects the salt and pepper specialist. "
             "A clean image would bypass the experts entirely through an identity route.",
    "soft": "Task three, the soft mixture of experts. Instead of picking one expert, a gate produces weights over the "
            "identity branch and the three specialists. For this blurred image, the blur expert gets most of the "
            "weight, and the output blends the branches. This model gave the best test results.",
    "soft2": "With rectangular occlusion the weights shift towards the occlusion expert. The weights always sum to "
             "one, and the balance loss during training stops the gate from collapsing onto a single expert.",
    "sketch": "Task four, the style conditioned conditional G A N. I choose a face photo and a sketch style, press "
              "generate, and the generator produces a pencil sketch in that style. The result can be downloaded.",
    "sketch2": "Changing the style to style three gives a different artist's rendering of the same face.",
    "sketch3": "And style one, on another face. The same generator serves all three styles, conditioned on a style "
               "label. On the test split the sketches reach an L one error of zero point one zero two.",
    "mlflow": "Finally, M L Flow tracking, started from the same Compose file with the tracking profile. Every "
              "training run and Optuna trial for the four tasks is recorded with its parameters and metrics.",
    "mlflow2": "Opening an experiment shows its runs with the logged metrics, so the reported results can be traced "
               "back to the exact training run.",
    "onnx": "All models were exported to ONNX and verified against the PyTorch versions, with a maximum difference "
            "below four times ten to the minus six. The backend only needs ONNX Runtime, so no GPU or PyTorch is "
            "needed to run the application.",
    "outro": "That completes the demo. The code, the models release, the report and the instructions are in the "
             "GitHub repository. Thank you.",
}


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


CAPTION_FIXES = [  # spoken spelling -> written form, so captions read normally
    ("Generative A I", "Generative AI"), ("S S I M", "SSIM"), ("P S N R", "PSNR"), ("M L Flow", "MLflow"),
    ("L one plus S S I M", "L1 + SSIM"), ("L one", "L1"), ("G A N", "GAN"), ("pix two pix", "pix2pix"),
    ("dash dash", "--"), ("ninety nine point six percent", "99.6%"), ("zero point six seven one", "0.671"),
    ("zero point seven two three", "0.723"), ("twenty four point seven nine", "24.79"),
    ("twenty six point two eight", "26.28"), ("zero point one zero two", "0.102"),
    ("four times ten to the minus six", "4e-6"), ("three thousand", "3000"), ("Task one", "Task 1"),
    ("Task two", "Task 2"), ("Task three", "Task 3"), ("Task four", "Task 4"), ("style three", "style 3"),
    ("style one", "style 1"), ("seven ONNX", "7 ONNX"), ("four tasks", "4 tasks"), ("three specialist", "3 specialist"),
    ("decibels", "dB"),
]


def caption_text(s):
    for a, b in CAPTION_FIXES:
        s = s.replace(a, b)
    return s


def write_srt(path, starts, dur):
    """One caption per sentence, timed in proportion to its length inside the narration clip of its scene."""
    import re
    n, lines = 1, []
    for k, text in NARRATION.items():
        sents = [s.strip() for s in re.split(r"(?<=[.!?:])\s+", text) if s.strip()]
        total = sum(len(s) for s in sents)
        t = starts[k]
        for s in sents:
            d = dur[k] * len(s) / total
            lines.append(f"{n}\n{srt_time(t)} --> {srt_time(t + d)}\n{caption_text(s)}\n")
            n += 1
            t += d
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def tts(text, wav, voice_dir=None):
    if voice_dir and (Path(voice_dir) / Path(wav).name).exists():  # pre-recorded / cloned voice for this scene
        shutil.copy(Path(voice_dir) / Path(wav).name, wav)
        with wave.open(str(wav)) as w:
            return w.getnframes() / w.getframerate()
    txt = Path(wav).with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$s.Rate = -1; $s.SetOutputToWaveFile('%s'); $s.Speak([IO.File]::ReadAllText('%s')); $s.Dispose()" % (wav, txt))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def card(page, title, body):
    page.set_content(f"""<body style="margin:0;background:#0f172a;color:#e2e8f0;font-family:Segoe UI,sans-serif;
    padding:60px"><h1 style="font-size:44px;margin:0 0 30px">{html.escape(title)}</h1>
    <pre style="font-size:20px;line-height:1.5;background:#020617;padding:24px;border-radius:12px;
    white-space:pre-wrap">{html.escape(body)}</pre></body>""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3000")
    ap.add_argument("--mlflow", default="http://localhost:5000")
    ap.add_argument("--out", default="demo/demo.mp4")
    ap.add_argument("--voice-dir", default=None, help="folder with <scene>.wav files to use instead of the TTS voice")
    ap.add_argument("--no-captions", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="demo_"))

    dur = {}
    for k, t in NARRATION.items():
        dur[k] = tts(t, work / f"{k}.wav", a.voice_dir)
        print(k, round(dur[k], 1), "s")
    ps_out = subprocess.run([DOCKER, "compose", "--profile", "tracking", "ps", "--format",
                             "table {{.Name}}\t{{.Service}}\t{{.Status}}\t{{.Ports}}"], capture_output=True, text=True,
                            cwd=ROOT).stdout
    health = subprocess.run(["curl", "-s", "localhost:3000/api/health"], capture_output=True, text=True).stdout
    ver = (ROOT / "models" / "onnx_verification.json").read_text()

    starts = {}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        ctx = b.new_context(viewport={"width": 1440, "height": 810}, record_video_dir=str(work),
                            record_video_size={"width": 1440, "height": 810})
        page = ctx.new_page()
        t0 = time.time()

        def begin(k):
            starts[k] = time.time() - t0
            return time.time() + dur[k] + 0.6

        def hold(until):
            left = until - time.time()
            if left > 0:
                page.wait_for_timeout(int(left * 1000))

        def scroll_slowly(until, step=80):
            while time.time() < until - 0.5:
                page.mouse.wheel(0, step)
                page.wait_for_timeout(2500)

        def open_page(h):
            page.goto(f"{a.url}/#{h}")
            page.reload()
            page.wait_for_selector("text=Backend online", timeout=15000)

        def run(corruption, level, button, wait_text, face=False, style=None, idx=1):
            sel = "img[alt^='face_']" if face else "img[alt^='pet_']"
            page.wait_for_selector(sel, timeout=15000)
            page.locator(sel).nth(idx).click()
            page.wait_for_timeout(700)
            if corruption:
                page.get_by_text(corruption, exact=True).click()
                page.wait_for_timeout(500)
                page.get_by_text(level, exact=True).click()
                page.wait_for_timeout(500)
            if style:
                page.get_by_text(style, exact=True).click()
                page.wait_for_timeout(500)
            page.get_by_role("button", name=button).click()
            page.wait_for_selector(f"text={wait_text}", timeout=30000)
            page.wait_for_timeout(1000)

        end = begin("intro")
        card(page, "Restoration-GAN", "Generative AI - Assignment 1\n\nTask 1  Universal denoising autoencoder\n"
             "Task 2  Classifier + specialist autoencoders (hard routing)\nTask 3  Soft mixture of experts\n"
             "Task 4  Style-conditioned face-to-sketch cGAN\n\nFastAPI + ONNX backend  |  React + Tailwind frontend  |  "
             "Docker Compose  |  MLflow + Optuna")
        hold(end)

        end = begin("pipeline")
        card(page, "Pipeline", "Task 1  corrupted image -> universal AE -> restored\n\n"
             "Task 2  image -> classifier -> {salt-pepper | blur | occlusion specialist}  (hard routing)\n"
             "        clean -> identity bypass\n\n"
             "Task 3  image -> gate (init from classifier) -> weighted sum of 3 experts + identity (soft MoE)\n\n"
             "Task 4  face photo + style label -> U-Net generator (PatchGAN trained) -> pencil sketch\n\n"
             "Optuna search per task, MLflow tracking, ONNX export + verification")
        hold(end)

        end = begin("docker")
        card(page, "docker compose up --build", ps_out + "\n\n/api/health:\n" + health)
        hold(end)

        end = begin("results")
        card(page, "Test results (measured)",
             "Classifier          accuracy 0.9963   macro-F1 0.9939\n"
             "Task 1 universal    21.23 dB / SSIM 0.723   (noisy input 22.33 dB / 0.671)\n"
             "Task 2 hard route   24.79 dB / SSIM 0.753   (oracle = predicted routing)\n"
             "Task 3 soft MoE     26.28 dB / SSIM 0.814\n"
             "Task 4 sketch cGAN  L1 0.102   15.91 dB   SSIM 0.493   (n = 1046)")
        hold(end)

        end = begin("universal")
        open_page("universal")
        run("Salt-and-pepper", "High", "Restore image", "Inference time")
        scroll_slowly(end)
        hold(end)
        end = begin("universal2")
        page.mouse.wheel(0, -2000)
        run("Gaussian blur", "Medium", "Restore image", "Inference time", idx=4)
        hold(end)

        end = begin("hard")
        open_page("hard")
        run("Rectangular occlusion", "Medium", "Restore image", "Classifier probabilities")
        scroll_slowly(end)
        hold(end)
        end = begin("hard2")
        page.mouse.wheel(0, -2000)
        run("Salt-and-pepper", "High", "Restore image", "Classifier probabilities", idx=2)
        scroll_slowly(end)
        hold(end)

        end = begin("soft")
        open_page("soft")
        run("Gaussian blur", "High", "Restore image", "Routing weights")
        scroll_slowly(end)
        hold(end)
        end = begin("soft2")
        page.mouse.wheel(0, -2000)
        run("Rectangular occlusion", "High", "Restore image", "Routing weights", idx=3)
        scroll_slowly(end)
        hold(end)

        end = begin("sketch")
        page.goto(f"{a.url}/#sketch")
        page.reload()
        run(None, None, "Generate sketch", "Download result", face=True, style="Style 2", idx=3)
        hold(end)
        end = begin("sketch2")
        page.get_by_text("Style 3", exact=True).click()
        page.get_by_role("button", name="Generate sketch").click()
        hold(end)
        end = begin("sketch3")
        page.locator("img[alt^='face_']").nth(5).click()
        page.get_by_text("Style 1", exact=True).click()
        page.get_by_role("button", name="Generate sketch").click()
        hold(end)

        end = begin("mlflow")
        page.goto(a.mlflow)
        page.wait_for_timeout(9000)
        hold(end)
        end = begin("mlflow2")
        page.goto(a.mlflow + "/#/experiments/18")
        page.wait_for_timeout(8000)
        hold(end)

        end = begin("onnx")
        card(page, "ONNX export and verification", ver)
        hold(end)

        end = begin("outro")
        card(page, "Restoration-GAN", "github.com/MusaxKhan/Restoration-GAN\n\nCode, ONNX models (release v1.0), IEEE "
             "report, README with execution instructions.")
        hold(end)
        page.wait_for_timeout(500)
        ctx.close()
        vid = next(work.glob("*.webm"))
        b.close()

    # audio track: each narration clip delayed to its scene start
    cmd = [FFMPEG, "-y", "-i", str(vid)]
    for k in NARRATION:
        cmd += ["-i", str(work / f"{k}.wav")]
    filt = "".join(f"[{i + 1}:a]adelay={int(starts[k] * 1000)}|{int(starts[k] * 1000)}[a{i}];"
                   for i, k in enumerate(NARRATION))
    filt += "".join(f"[a{i}]" for i in range(len(NARRATION))) + f"amix=inputs={len(NARRATION)}:normalize=0[aout]"
    srt = out.with_suffix(".srt")
    write_srt(srt, starts, dur)
    vf = []
    if not a.no_captions:  # burn the captions into the picture (also kept as a separate .srt for YouTube)
        shutil.copy(srt, work / "caps.srt")
        vf = ["-vf", "subtitles=caps.srt:force_style='FontSize=18,Outline=2,MarginV=18'"]
    cmd += ["-filter_complex", filt, "-map", "0:v", "-map", "[aout]", *vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out.resolve())]
    subprocess.run(cmd, check=True, cwd=work)
    print("saved", out, "narration ~", round(sum(dur.values()) / 60, 1), "min")


if __name__ == "__main__":
    main()
