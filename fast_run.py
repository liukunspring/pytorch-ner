import pytorch_ner



def convert_token_to_bert():
    token_seq, label_seq  = pytorch_ner.prepare_conll_data_format_bert('data/conll2003/train.txt',sep=' ')
    for index,_ in enumerate(token_seq):
        token_list=token_seq[index]
        lable_list=label_seq[index]
        line1 = ""
        line2 = ""
        for word, full_label in zip(token_list, lable_list):
            #full_label = label_names[label]
            max_length = max(len(word), len(full_label))
            line1 += word + " " * (max_length - len(word) + 1)
            line2 += full_label + " " * (max_length - len(full_label) + 1)

        print(line1)
        print(line2)
    
     
    
if __name__=='__main__':
    pytorch_ner.train(path_to_config="config.yaml")