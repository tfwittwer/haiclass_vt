"""Check that an interrupted run resumes instead of restarting.

Weights alone are not enough. The failure this guards against is silent:
reload only the model and OneCycleLR restarts its warm-up at full learning
rate, undoing a converged run. Run: python test_resume.py
"""
import numpy as np
import torch


def build():
    m = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=100, pct_start=0.05)
    return m, opt, sched, torch.amp.GradScaler(enabled=False)


def test_resume(tmp="/tmp/haiclass_resume_test.pt"):
    m, opt, sched, scaler = build()
    rng = np.random.default_rng(42)
    for _ in range(40):
        m(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        sched.step()
    want_lr = opt.param_groups[0]["lr"]
    # Snapshot the generator, then draw: the resumed run must produce this same
    # next shift, i.e. carry on the sequence rather than repeat or skip a draw.
    rng_state = rng.bit_generator.state
    want_shift = rng.uniform(0, 30, 2)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state,
                "torch_rng": torch.get_rng_state(), "epoch": 40, "best_miou": 0.4786}, tmp)

    ck = torch.load(tmp, weights_only=False)
    m2, opt2, sched2, scaler2 = build()
    m2.load_state_dict(ck["model"])
    opt2.load_state_dict(ck["opt"])
    sched2.load_state_dict(ck["sched"])
    scaler2.load_state_dict(ck["scaler"])
    rng2 = np.random.default_rng(0)
    rng2.bit_generator.state = ck["rng"]
    torch.set_rng_state(ck["torch_rng"].cpu())

    assert sched2.last_epoch == 40, f"schedule position lost: {sched2.last_epoch}"
    assert np.isclose(opt2.param_groups[0]["lr"], want_lr), "LR restarted the warm-up"
    assert opt2.param_groups[0]["lr"] < 3e-4, "LR is at the warm-up peak, decay was lost"
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"], \
        "Adam moment step count lost"
    # same block partition as the uninterrupted run would have drawn
    assert np.allclose(rng2.uniform(0, 30, 2), want_shift), "data-order RNG not restored"
    assert ck["epoch"] + 1 == 41 and ck["best_miou"] == 0.4786
    print(f"ok: resumes at epoch 41, lr={opt2.param_groups[0]['lr']:.3e}")


if __name__ == "__main__":
    test_resume()
