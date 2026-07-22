import tiktoken
import torch
class Tokenizier:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base") # tokenizer
        self.special_list = {} # hold all unique charcaters that arent letters or numbers 
        self.base_name = "Custom_Encoding"
        self.encoder = None
        self.rebuild()
    def is_image(self,text):
        if "CFUI" in text:
            return False
        else:
            return True
    
    def rebuild(self): #to rebuild encoding for added special tokens 
        all_specials = {
            **self.tokenizer._special_tokens,
            **self.special_list
        }
        self.encoder = tiktoken.Encoding(
            name = f"{self.base_name}_dynamic",
            pat_str = self.tokenizer._pat_str,
            mergeable_ranks=self.tokenizer._mergeable_ranks,
            special_tokens=all_specials
        )

    def special_encoding(self, text):
        if text in self.tokenizer._special_tokens or text in self.special_list:
            return 
        else:
            next_id = self.tokenizer.n_vocab + len(self.special_list)
            self.special_list[text] = next_id
        self.rebuild()
        self.encoder.encode(text, allowed_specials=all)
    '''
    def encode_and_add(self, text):
        ### split and store raw text in vocab
        text = text.split()
        text = sorted(set(text))
        ## returns and stores the key values only for comparison
        keys = self.vocab_list.keys()
        ## loops and adds and key not already in the diction into both the vocab(uncoded) and the token_list(coded)
        for i in range(len(text)):
            if text[i] not in keys:
                if not self.is_english(text[i]):
                    self.special_encoding(text[i])
                else:
                    self.vocab_list.update({text[i] : len(self.vocab_list) + 1})
                    ####Encode always returns a list do you must put [0] or the index to get to the actual value and no tthe list itself
                    self.token_list.update({self.tokenizer.encode(text[i])[0]: len(self.token_list) + 1})  
        new = list(self.token_list.keys())  
        return new 
    '''
    def decodes(self, text):
        if hasattr(text, "tolist"):
            text = text.tolist()
        elif isinstance(text, torch.Tensor): 
            text = text.cpu().numpy().tolist()
        token_ids = [item for item in text if isinstance(item, int)]
        return self.encoder.decode(token_ids)
    def encode(self, text, allowed_special="all"):   
        if isinstance(text, torch.Tensor):
            text = text.cpu().numpy().tolist()  
        if isinstance(text, list):
            for word in text:
                if hasattr(self, "is_image") and self.is_image(word):
                    self.special_encoding(word)
            text = " ".join(text)
        return self.encoder.encode(text, allowed_special=allowed_special)
    def get_vocab_length(self):
        return len(self.vocab_list)
    def get_token_list_length(self):
        return len(self.token_list)
'''
Testing 

import sys
import io
sys.stdout  = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
token = Tokenizier()
sample_text = "This is a 테스트용 텍스트 but"
sample_text_2 = " i hate everything here lmao"
print(token.encode(sample_text))      
print(token.decodes(token.encode(sample_text))) 
'''

