import argparse
from fabric_pipeline.common import guarded_main, load_config
from fabric_pipeline.data import audit_source, prepare_original


def main():
    parser = argparse.ArgumentParser(description="Audit original CHENAB and prepare immutable-input training copies")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    audit_source(cfg)
    prepare_original(cfg)


if __name__ == "__main__":
    guarded_main(main)
