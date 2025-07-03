import numpy as np
import cv2
import triton_python_backend_utils as pb_utils
import re
import os
states=[
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB", "AN", "CH",
    "DN", "DL", "JK", "LA", "LD", "PY"
]
digits=["0","1","2","3","4","5","6","7","8","9"]
chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']

char_map = {
    '0': 1,  '1': 2,  '2': 3,  '3': 4,  '4': 5,
    '5': 6,  '6': 7,  '7': 8,  '8': 9,  '9': 10,
    ':': 11, ';': 12, '<': 13, '=': 14, '>': 15,
    '?': 16, '@': 17, 'A': 18, 'B': 19, 'C': 20,
    'D': 21, 'E': 22, 'F': 23, 'G': 24, 'H': 25,
    'I': 26, 'J': 27, 'K': 28, 'L': 29, 'M': 30,
    'N': 31, 'O': 32, 'P': 33, 'Q': 34, 'R': 35,
    'S': 36, 'T': 37, 'U': 38, 'V': 39, 'W': 40,
    'X': 41, 'Y': 42, 'Z': 43, '[': 44, '\\': 45,
    ']': 46, '^': 47, '_': 48, '`': 49, 'a': 50,
    'b': 51, 'c': 52, 'd': 53, 'e': 54, 'f': 55,
    'g': 56, 'h': 57, 'i': 58, 'j': 59, 'k': 60,
    'l': 61, 'm': 62, 'n': 63, 'o': 64, 'p': 65,
    'q': 66, 'r': 67, 's': 68, 't': 69, 'u': 70,
    'v': 71, 'w': 72, 'x': 73, 'y': 74, 'z': 75,
    '{': 76, '|': 77, '}': 78, '~': 79, '!': 80,
    '"': 81, '#': 82, '$': 83, '%': 84, '&': 85,
    "'": 86, '(': 87, ')': 88, '*': 89, '+': 90,
    ',': 91, '-': 92, '.': 93, '/': 94, ' ':96
}
def KapiDecoder(preds):
    print(preds.shape)
    # preds_idx = np.argmax(preds, axis=2)
    # print(preds_idx.shape)
    # pos=[]
    # print(preds_idx[0][0])
    # for i in range(preds_idx.shape[1]):
    #     if(preds_idx[0][i]==0):
    #         continue
    #     else:
    #         pos.append(i)
    # print(pos)
    final_ans=[]
    for i in range(preds.shape[0]):
        preds_idx=np.argmax(preds[i],axis=1)
        print(preds_idx.shape)
        tot_prob=0
        pos=np.array([])
        for j in range(preds_idx.shape[0]):
            if preds_idx[j]==0 or preds_idx[j]==96:
                continue
            else:
                #print("yes")
                pos=np.append(pos,j)
        print("pos is")
        print(pos)
        pos=pos.astype(np.uint8)
        # print(pos[0])
        # print(preds[0][int(pos[0])][32])
        if(pos.shape[0]<9 or pos.shape[0]>12):
            continue
        ans=""
        max_probs1=-1e9
        max_probs2=-1e9
        s_1=""
        s_2=""
        
        # calculating the probavility of possible state codes in first and second index
        for s in states:
            first=s[0]
            first=char_map[s[0]]
            first=preds[i][int(pos[0])][first]
            second=s[1]
            second=char_map[s[1]]
            second=preds[i][int(pos[1])][second]
            probs_1=first+second
            if probs_1>max_probs1:
                max_probs1=probs_1
                s_1=s
            
        #calculating the same for second and third
        for s in states:
            first=s[0]
            first=char_map[s[0]]
            first=preds[i][pos[1]][first]
            second=s[1]
            second=char_map[s[1]]
            second=preds[i][pos[2]][second]
            probs_2=first+second
            if probs_2>max_probs2:
                max_probs2=probs_2
                s_2=s
        first_t=0
        if(max_probs1>=max_probs2):
            ans+=s_1
            first_t=1
            tot_prob=max_probs1
        else:
            ans+=s_2
            tot_probs=max_probs2
        print(f"ans is {ans}")
        #now find the next 2 digits
        last_parsed=0
        next_nums=[]
        if first_t:
            #next_nums=[2,3]
            last_parsed=1
        else:
            #next_nums=[3,4]
            last_parsed=2
        next_nums=[last_parsed+1,last_parsed+2]
        for n in next_nums:
            idx=pos[n]
            max_prob=-1e9
            max_d=""
            for d in digits:
                char_idx=char_map[d]
                prob=preds[i][idx][char_idx]
                if prob>max_prob:
                    max_prob=prob
                    max_d=d
            tot_prob+=max_prob
            ans+=max_d
        print(f"ans is {ans}")
        last_parsed+=2
        while(last_parsed+5<pos.shape[0]):
            idx=pos[last_parsed+1]
            max_prob=-1e9
            max_c=""
            for c in chars:
                char_idx=char_map[c]
                prob=preds[i][idx][char_idx]
                if prob>max_prob:
                    max_prob=prob
                    max_c=c
            ans+=max_c
            tot_prob+=max_prob
            last_parsed+=1
        print(f"ans is {ans}")
        while(last_parsed+1<pos.shape[0]):
            idx=pos[last_parsed+1]
            max_prob=-1e9
            max_d=""
            for d in digits:
                char_idx=char_map[d]
                prob=preds[i][idx][char_idx]
                if prob>max_prob:
                    max_prob=prob
                    max_d=d
            tot_prob+=max_prob
            ans+=max_d
            last_parsed+=1
        print(f"ans is {ans}")
        print(f"the confidence is {tot_prob/len(ans)}")
        final_ans.append(ans)
    return final_ans
class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, selection):
        """
        Group the decoded characters and record the corresponding decoded positions.

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types of grouping words:
                        - 'cn': continuous chinese characters (e.g., 你好啊)
                        - 'en&num': continuous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
                        The remaining characters in text are treated as separators between groups (e.g., space, '(', ')', etc.).
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection == True)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            elif bool(re.search("[a-zA-Z0-9]", char)):
                c_state = "en&num"
            else:
                c_state = "splitter"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ):  # grouping floating number
                c_state = "en&num"
            if (
                char == "-" and state == "en&num"
            ):  # grouping word with '-', such as 'state-of-the-art'
                c_state = "en&num"

            if state == None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            #print("Selection", text_index[batch_idx][selection])
            #print("chars size", len(self.character))
            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, return_word_box=False, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
            return_word_box=return_word_box,
        )
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character
class TritonPythonModel:
    def initialize(self, args):
        self.use_space_char=True
        self.rec_char_dict_path = "/models/PaddleOCR-Postprocessing/1/en_dict.txt"
        self.decoder=CTCLabelDecode(character_dict_path=self.rec_char_dict_path,use_space_char=self.use_space_char)
    def execute(self, requests):
        request = requests[0]  # Expecting only one request due to max_batch_size: 0
        #print("Received request:", request)
        # Inputs
        output = pb_utils.get_input_tensor_by_name(request, "fetch_name_0").as_numpy()
        # wh_ratio_list = pb_utils.get_input_tensor_by_name(request, "wh_ratio_list").as_numpy()
        # sorted_indices = pb_utils.get_input_tensor_by_name(request, "sorted_indices").as_numpy()
        # max_wh_ratio = pb_utils.get_input_tensor_by_name(request, "max_wh_ratio").as_numpy()
        #
        # print("Output shape:", output.shape)
        # rec_result = [1,2,3]
        # Run decoder
        # rec_result = self.decoder(
        #     output,
        #     return_word_box=False,
        #     wh_ratio_list=wh_ratio_list,
        #     max_wh_ratio=max_wh_ratio
        # )
        final_ans=KapiDecoder(output)
        
        #print("Decoded results:", rec_result)
        # Apply alphanumeric filter (remove all non-alphanumeric characters including spaces)
        print("final ans ",final_ans)
        # print("rec results",rec_result)
        # filtered_result = [re.sub(r'[^A-Za-z0-9]', '', rec[0]) for rec in rec_result]

        output_tensor = pb_utils.Tensor(
            "OUTPUT_TEXT",
            np.array(final_ans, dtype=object)
        )

        response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
        return [response]