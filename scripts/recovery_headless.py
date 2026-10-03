"""One irreversible request reservation per recovery slot, then the frozen guard."""
from shinka_crl.adaptive_recovery import provider_main


if __name__ == "__main__":
    raise SystemExit(provider_main())
