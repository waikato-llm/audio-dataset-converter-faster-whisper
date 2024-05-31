# fw-transcribe

* accepts: adc.api.SpeechData
* generates: adc.api.SpeechData

Generates transcriptions for the audio files passing through.

```
usage: fw-transcribe [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     [-N LOGGER_NAME] [-m MODEL_SIZE] [-d DEVICE]
                     [-c COMPUTE_TYPE] [-b BEAM_SIZE]

Generates transcriptions for the audio files passing through.

optional arguments:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -m MODEL_SIZE, --model_size MODEL_SIZE
                        The size of the whisper model to use, e.g., 'base' or
                        'large-v3' (default: base)
  -d DEVICE, --device DEVICE
                        The device to run on, e.g., 'cuda' or 'cpu' (default:
                        cpu)
  -c COMPUTE_TYPE, --compute_type COMPUTE_TYPE
                        The compute type to use, e.g., 'float16' or 'int8'
                        (default: int8)
  -b BEAM_SIZE, --beam_size BEAM_SIZE
                        The beam size to use for decoding (default: 5)
```
