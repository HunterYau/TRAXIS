"""Existing tiny bonded-model checks through the separate batched adapter."""
def main():
    from traxis import gpu_local_aluminum as reference
    from traxis.gpu_local_aluminum_batched import build_system
    from examples import local_aluminum_check
    original=reference.build_system
    try:
        # Process-local factory selection only; no solver source is modified.
        reference.build_system=build_system
        local_aluminum_check.main()
    finally:reference.build_system=original
if __name__=='__main__':main()
