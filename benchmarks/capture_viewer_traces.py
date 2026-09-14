"""Capture compatible accuracy-policy modes for the explanatory viewer."""
import argparse
import importlib.metadata
import json
import multiprocessing
import platform
import subprocess
from pathlib import Path
from threadpoolctl import threadpool_limits
from evaluate_adaptive import worker


def capture(connection, name, seed, method):
    try:
        connection.send(worker(name,seed,method,.25,policy="accuracy"))
    finally:
        connection.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="examples/viewer_traces.json");args=parser.parse_args()
    rows=[]
    for name in ["iris","wine","breast_cancer","xor3","easy","noisy"]:
        for seed in [0,1,2]:
            for method in ["highs","direct","gap"]:
                context=multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
                receive,send=context.Pipe(duplex=False)
                proc=context.Process(target=capture,args=(send,name,seed,method))
                with threadpool_limits(limits=1):
                    proc.start();send.close();rows.append(receive.recv());proc.join()
                receive.close()
    payload=dict(schema_version=1,revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                 platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ["numpy","pandas","pulp","highspy","scikit-learn"]},
                 budget_note="0.25-second cooperative allowance; all three modes use the established accuracy acceptance policy. Imports excluded, fresh isolated process per fit, imports preloaded on fork platforms.",results=rows)
    Path(args.output).write_text(json.dumps(payload,indent=2)+"\n")


if __name__=="__main__":main()
