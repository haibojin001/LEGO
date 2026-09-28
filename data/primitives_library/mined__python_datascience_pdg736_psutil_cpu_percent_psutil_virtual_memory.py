# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg736::psutil.cpu_percent+psutil.virtual_memory
# name: psutil_primitive
# summary: Uses psutil.cpu_percent, psutil.virtual_memory across 2 repos
# anchor_symbols: ['psutil.cpu_percent', 'psutil.virtual_memory']
# observed in 2 repos: ['DeepWisdom__AutoDL', 'polyaxon__traceml']...

# --- from DeepWisdom__AutoDL::AutoDL_scoring_program/score.py::end_file_generated ---
def end_file_generated(prediction_dir):
  """Check if ingestion is still alive by checking if the file 'end.txt'
  is generated in the folder of predictions.
  """
  end_filepath =  os.path.join(prediction_dir, 'end.txt')
  logger.debug("CPU usage: {}%".format(psutil.cpu_percent()))
  logger.debug("Virtual memory: {}".format(psutil.virtual_memory()))
  return os.path.isfile(end_filepath)

# --- from polyaxon__traceml::traceml/traceml/processors/psutil_processor.py::query_psutil ---
def query_psutil() -> Dict:
    results = {}
    try:
        # psutil <= 5.6.2 did not have getloadavg:
        if hasattr(psutil, "getloadavg"):
            results["load"] = psutil.getloadavg()[0]
        else:
            # Do not log an empty metric
            pass
    except OSError:
        pass
    vm = psutil.virtual_memory()
    results["cpu"] = psutil.cpu_percent(interval=None)
    results["memory"] = vm.percent
    return results
