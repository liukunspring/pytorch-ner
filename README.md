### About
Pipeline for training NER models using PyTorch.<br/>
ONNX export supported.

### Usage
The user interface consists of only one file *config.yaml*.<br/>
Change *config.yaml* to create the desired configuration and start the pipeline with the following command:
```
python main.py --config config.yaml
```
If *--config* argument is not specified, then used config.yaml.

To export trained model to ONNX use config.yaml:
```
save:
  export_onnx: True
```

### Data Format:
Text file containing separated tokens and labels on each line. Sentences are separated by empty line.
Labels should already be in necessary format, e.g. IO, BIO, BILUO, ...

Example:
```
token_11    label_11
token_12    label_12

token_21    label_21
token_22    label_22
token_23    label_23

...
```

### Models
List of implemented models:
- [x] BiLTSM
- [ ] BiLTSMCRF
- [ ] BiLTSMAttn
- [ ] BiLTSMAttnCRF
- [ ] BiLTSMCNN
- [ ] BiLTSMCNNCRF
- [ ] BiLTSMCNNAttn
- [ ] BiLTSMCNNAttnCRF

<!--
# TODO: add model results
-->
