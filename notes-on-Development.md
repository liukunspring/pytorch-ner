## 2025年1月4日

### 2025-1-4 01:01:37
尝试把bert的tokenizer把标签的分词引入到token中，在相同的参数下，查看一下训练的参数。
仅仅有一点效果但是实际上验证数据集的结果还是很差。
```
pytorch-ner-train - INFO - epoch [10/10]

100%|███████████████████████████| 59/59 [00:34<00:00,  1.69it/s] 
pytorch-ner-train - INFO - train loss: 0.02864180596829471       
pytorch-ner-train - INFO - train f1 B-LOC: 0.9778825198142723    
pytorch-ner-train - INFO - train f1 B-MISC: 0.9589228035708544   
pytorch-ner-train - INFO - train f1 B-ORG: 0.9677206833121432    
pytorch-ner-train - INFO - train f1 B-PER: 0.9817644033483944    
pytorch-ner-train - INFO - train f1 I-LOC: 0.9815364842345262    
pytorch-ner-train - INFO - train f1 I-MISC: 0.9577245494110119   
pytorch-ner-train - INFO - train f1 I-ORG: 0.9806564498346755    
pytorch-ner-train - INFO - train f1 I-PER: 0.9940951407159063    
pytorch-ner-train - INFO - train f1 O: 0.9977296706061486        
pytorch-ner-train - INFO - train f1-weighted: 0.9940752816963498 
pytorch-ner-train - INFO -

100%|██████████████████████| 3465/3465 [00:14<00:00, 233.27it/s] 
pytorch-ner-train - INFO - valid loss: 0.24123969396726602       
pytorch-ner-train - INFO - valid f1 B-LOC: 0.3134877292020149    
pytorch-ner-train - INFO - valid f1 B-MISC: 0.13330859616573904  
pytorch-ner-train - INFO - valid f1 B-ORG: 0.18506630935202362   
pytorch-ner-train - INFO - valid f1 B-PER: 0.22662574529265553   
pytorch-ner-train - INFO - valid f1 I-LOC: 0.09719951525146331   
pytorch-ner-train - INFO - valid f1 I-MISC: 0.057661543202668746 
pytorch-ner-train - INFO - valid f1 I-ORG: 0.11963696431755616   
pytorch-ner-train - INFO - valid f1 I-PER: 0.22628938780133434   
pytorch-ner-train - INFO - valid f1 O: 0.9576298373863217        
pytorch-ner-train - INFO - valid f1-weighted: 0.9370495036143184 
pytorch-ner-train - INFO -

pytorch-ner-train - INFO - test f1 B-PER: 0.17580660094588785
pytorch-ner-train - INFO - test f1 I-LOC: 0.06524657274725154
pytorch-ner-train - INFO - test f1 I-MISC: 0.03216343646699334
pytorch-ner-train - INFO - test f1 I-ORG: 0.12988257108432308
pytorch-ner-train - INFO - test f1 I-PER: 0.18494738005076525
pytorch-ner-train - INFO - test f1 O: 0.9466361428285641
pytorch-ner-train - INFO - test f1-weighted: 0.908866438510154
pytorch-ner-train - INFO -
```


### 2025-1-4 00:01:43
下载代码之后发现实际上lstm上训练的效果并不好，主要是验证数据集上valid f1 非O级别的训练数据集太差了。

数据如下，所以尝试增加一些方法提升训练训练真正的验证数据集的f1数值
```
pytorch-ner-train - INFO - epoch [10/10]

100%|███████████████████████████████████████████| 59/59 [00:22<00:00,  2.60it/s] 
pytorch-ner-train - INFO - train loss: 0.019507784791038197
pytorch-ner-train - INFO - train f1 B-LOC: 0.9869133981875062
pytorch-ner-train - INFO - train f1 B-MISC: 0.9775327590085755
pytorch-ner-train - INFO - train f1 B-ORG: 0.9801605670927993
pytorch-ner-train - INFO - train f1 B-PER: 0.9923895681493464
pytorch-ner-train - INFO - train f1 I-LOC: 0.9812405286675006
pytorch-ner-train - INFO - train f1 I-MISC: 0.9725741301993837
pytorch-ner-train - INFO - train f1 I-ORG: 0.983661932030829
pytorch-ner-train - INFO - train f1 I-PER: 0.9960733790775697
pytorch-ner-train - INFO - train f1 O: 0.9990375500244498
pytorch-ner-train - INFO - train f1-weighted: 0.9968875741073576
pytorch-ner-train - INFO -

100%|██████████████████████████████████████| 3465/3465 [00:13<00:00, 266.20it/s] 
pytorch-ner-train - INFO - valid loss: 0.25402646042806293
pytorch-ner-train - INFO - valid f1 B-LOC: 0.3066183551897838
pytorch-ner-train - INFO - valid f1 B-MISC: 0.14324285943333562
pytorch-ner-train - INFO - valid f1 B-ORG: 0.1975094794575314
pytorch-ner-train - INFO - valid f1 B-PER: 0.24019836024165028
pytorch-ner-train - INFO - valid f1 I-LOC: 0.044983164983164986
pytorch-ner-train - INFO - valid f1 I-MISC: 0.043302115458978205
pytorch-ner-train - INFO - valid f1 I-ORG: 0.07355778981320106
pytorch-ner-train - INFO - valid f1 I-PER: 0.177110432127976
pytorch-ner-train - INFO - valid f1 O: 0.9396070396871883
pytorch-ner-train - INFO - valid f1-weighted: 0.9255134635379759
pytorch-ner-train - INFO -

100%|██████████████████████████████████████| 3683/3683 [00:13<00:00, 273.24it/s] 
pytorch-ner-train - INFO - test loss: 0.44863772296594595
pytorch-ner-train - INFO - test f1 B-LOC: 0.23467803201715776
pytorch-ner-train - INFO - test f1 B-MISC: 0.09712514383977865
pytorch-ner-train - INFO - test f1 B-ORG: 0.23840004988525215
pytorch-ner-train - INFO - test f1 B-PER: 0.1792425015247627
pytorch-ner-train - INFO - test f1 I-LOC: 0.028038736537243186
pytorch-ner-train - INFO - test f1 I-MISC: 0.0250328622715264
pytorch-ner-train - INFO - test f1 I-ORG: 0.08700213440354948
pytorch-ner-train - INFO - test f1 I-PER: 0.13583946271487812
pytorch-ner-train - INFO - test f1 O: 0.8989890977221824
pytorch-ner-train - INFO - test f1-weighted: 0.8695545071930769
pytorch-ner-train - INFO -
```