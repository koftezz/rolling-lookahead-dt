"""Generate a dependency-free, measured-results viewer from experiment JSON."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="examples/viewer_traces.json")
    parser.add_argument("--output",default="examples/trace_viewer.html")
    args=parser.parse_args()
    payload=json.loads(Path(args.input).read_text())
    template=Path(__file__).with_name("trace_viewer_template.html").read_text()
    encoded=json.dumps(payload).replace("</", r"<\/")
    Path(args.output).write_text(template.replace("__MEASURED_DATA__",encoded))


if __name__=="__main__":
    main()
