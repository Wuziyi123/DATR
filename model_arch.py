import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple
import copy
from transformers import BertModel, BertConfig, BertTokenizer, AutoModel


# 设置NLTK数据路径
nltk_data_paths = [
    # '/root/nltk_data',
    '/root/miniconda3/envs/WCA/nltk_data',
]
for path in nltk_data_paths:
    if os.path.exists(path):
        nltk.data.path.append(path)

# 确保下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nltk.download('punkt_tab')


class TextAugmenter:
    """基于WordNet的文本增强器，用于生成负样本"""
    def __init__(self, augmentation_prob: float = 0.5):
        self.augmentation_prob = augmentation_prob
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 定义语义角色类别
        self.object_words = {'NN', 'NNS', 'NNP', 'NNPS'}  # 名词
        self.attribute_words = {'JJ', 'JJR', 'JJS'}  # 形容词
        self.relation_words = {'IN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}  # 介词和动词
        self.color_words = {'red', 'blue', 'green', 'yellow', 'orange', 'purple',
                            'pink', 'brown', 'black', 'white', 'gray', 'grey'}

    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        """获取单词的同义词"""
        synonyms = set()

        # 映射NLTK词性标签到WordNet词性
        pos_map = {
            'NN': wordnet.NOUN, 'NNS': wordnet.NOUN,
            'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
            'VB': wordnet.VERB, 'VBD': wordnet.VERB, 'VBG': wordnet.VERB,
            'VBN': wordnet.VERB, 'VBP': wordnet.VERB, 'VBZ': wordnet.VERB,
            'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV
        }

        wordnet_pos = pos_map.get(pos) if pos else None

        for syn in wordnet.synsets(word, pos=wordnet_pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word and len(synonym.split()) == 1:
                    synonyms.add(synonym)

        return list(synonyms)

    def identify_semantic_roles(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """识别文本中的语义角色"""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        semantic_roles = {
            'objects': [],  # (word, index)
            'attributes': [],  # (word, index)
            'relations': [],  # (word, index)
            'colors': []  # (word, index)
        }

        for i, (word, pos) in enumerate(pos_tags):
            word_lower = word.lower()

            if pos in self.object_words:
                semantic_roles['objects'].append((word, i))

            if pos in self.attribute_words:
                semantic_roles['attributes'].append((word, i))

            if pos in self.relation_words:
                semantic_roles['relations'].append((word, i))

            if word_lower in self.color_words:
                semantic_roles['colors'].append((word, i))

        return semantic_roles

    def synonym_replacement(self, text: str, roles: Dict) -> str:
        """同义词替换增强"""
        tokens = word_tokenize(text)

        # 选择要替换的单词（优先替换对象和属性词）
        replace_candidates = roles['objects'] + roles['attributes']
        if not replace_candidates:
            replace_candidates = roles['relations'] + roles['colors']

        if not replace_candidates:
            return text  # 没有可替换的单词

        # 随机选择要替换的单词
        word_to_replace, idx = random.choice(replace_candidates)
        pos_tag = nltk.pos_tag([word_to_replace])[0][1]

        # 获取同义词
        synonyms = self.get_synonyms(word_to_replace, pos_tag)
        if not synonyms:
            return text  # 没有同义词

        # 随机选择一个同义词
        new_word = random.choice(synonyms)
        tokens[idx] = new_word

        return ' '.join(tokens)

    def random_deletion(self, text: str, roles: Dict) -> str:
        """随机删除增强"""
        tokens = word_tokenize(text)

        if len(tokens) <= 6:
            return text  # 句子太短，不删除

        # 优先删除属性和关系词
        delete_candidates = roles['attributes'] + roles['relations']
        if not delete_candidates:
            delete_candidates = [(word, i) for i, word in enumerate(tokens)]

        # 随机选择要删除的单词
        word_to_delete, idx = random.choice(delete_candidates)
        del tokens[idx]

        return ' '.join(tokens)

    def subject_predicate_swap(self, text: str, roles: Dict) -> str:
        """主谓关系互换增强"""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        # 寻找主语和谓语
        subjects = []  # (word, index)
        predicates = []  # (word, index)

        for i, (word, pos) in enumerate(pos_tags):
            if pos in {'NN', 'NNS', 'NNP', 'NNPS'}:  # 名词可能是主语
                subjects.append((word, i))
            elif pos in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:  # 动词可能是谓语
                predicates.append((word, i))

        # 需要至少一个主语和一个谓语
        if not subjects or not predicates:
            return text

        # 随机选择一个主语和一个谓语
        subject, subj_idx = random.choice(subjects)
        predicate, pred_idx = random.choice(predicates)

        # 互换位置（简化处理，实际可能需要更复杂的语法调整）
        if subj_idx < pred_idx:
            tokens[subj_idx] = predicate
            tokens[pred_idx] = subject

        return ' '.join(tokens)

    def augment_text(self, text: str) -> str:
        """应用文本增强"""
        # 识别语义角色
        roles = self.identify_semantic_roles(text)

        # 随机选择一种增强方法
        augmentation_methods = [
            self.synonym_replacement,
            self.random_deletion,
            # self.subject_predicate_swap
        ]

        method = random.choice(augmentation_methods)
        augmented_text = method(text, roles)

        return augmented_text


class GradientModulationLayer(nn.Module):
    """梯度调制层，用于控制噪声样本对训练的影响"""

    def __init__(self, hidden_size: int, modulation_strength: float = 0.1):
        super().__init__()
        self.modulation_strength = modulation_strength
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor, is_augmented: bool = False) -> torch.Tensor:
        """
        前向传播

        参数:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            is_augmented: 是否是增强样本

        返回:
            调制后的隐藏状态
        """
        if not is_augmented:
            return hidden_states  # 对原始样本不做处理

        # 计算注意力权重
        attention_weights = self.attention_net(hidden_states)  # [batch_size, seq_len, 1]

        # 调制隐藏状态
        modulated_states = hidden_states * (1 - self.modulation_strength * attention_weights)

        return modulated_states


class EnhancedTextEncoder(nn.Module):
    """增强的文本编码器，集成负样本增强和梯度调制，适配原有TextEncoder特性"""
    def __init__(self, embed_dim: int = 512, augmentation_prob: float = 0.3, model_path: str = "./my_bert"):
        super().__init__()
        self.embed_dim = embed_dim
        self.augmentation_prob = augmentation_prob

        # 加载预训练的BERT模型（离线加载）
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)

        # 冻结BERT所有参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 解冻最后3层
        for layer in self.bert.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 文本增强器
        self.augmenter = TextAugmenter(augmentation_prob)

        # 梯度调制层
        # self.gradient_modulation = GradientModulationLayer(self.bert.config.hidden_size)

        # 投影层（与原有TextEncoder保持一致）
        self.pro_lo = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embed_dim),
            nn.BatchNorm1d(77),  # 保持序列长度维度
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pro_go = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )

        self.init_weights()

    def init_weights(self):
        """初始化权重（与原有TextEncoder保持一致）"""
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                original_texts: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，适配原有TextEncoder的输入输出格式

        参数:
            input_ids: 输入ID [batch_size, num_texts, seq_len]
            attention_mask: 注意力掩码 [batch_size, num_texts, seq_len]
            original_texts: 原始文本列表，用于增强（可选）

        返回:
            global_text_feats: 全局文本特征 [batch_size, num_texts, embed_dim]
            local_text_feats: 局部文本特征 [batch_size, num_texts, seq_len, embed_dim]
        """
        batch_size, num_texts, seq_len = input_ids.shape

        # 检查是否需要应用文本增强
        apply_augmentation = self.training and original_texts is not None

        if apply_augmentation:
            # 对部分样本应用文本增强
            augmented_texts = []
            is_augmented = []

            # 展平批次和文本数量维度
            flat_texts = []
            for i in range(batch_size):
                for j in range(num_texts):
                    if j == 0:  # 只对第一个文本进行增强（训练时每个图像只有一个文本）
                        if random.random() < self.augmentation_prob:
                            augmented_text = self.augmenter.augment_text(original_texts[i])
                            augmented_texts.append(augmented_text)
                            is_augmented.append(True)
                        else:
                            augmented_texts.append(original_texts[i])
                            is_augmented.append(False)
                    else:
                        augmented_texts.append(original_texts[i])
                        is_augmented.append(False)

            # 对增强后的文本重新进行tokenization
            augmented_inputs = self.tokenizer(
                augmented_texts,
                padding='max_length',
                truncation=True,
                max_length=seq_len,
                return_tensors='pt'
            )

            # 将设备设置为与input_ids相同
            augmented_inputs = {k: v.to(input_ids.device) for k, v in augmented_inputs.items()}

            # 使用增强后的输入
            input_ids = augmented_inputs['input_ids'].view(batch_size, num_texts, seq_len)
            attention_mask = augmented_inputs['attention_mask'].view(batch_size, num_texts, seq_len)
            is_augmented_tensor = torch.tensor(is_augmented, device=input_ids.device, dtype=torch.bool)
        else:
            is_augmented_tensor = torch.zeros(batch_size * num_texts, device=input_ids.device, dtype=torch.bool)

        # 展平批次和文本数量维度
        flat_input_ids = input_ids.view(batch_size * num_texts, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_texts, seq_len)

        # 获取BERT输出
        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch_size * num_texts, seq_len, hidden_size]

        # 应用梯度调制
        # modulated_states = self.gradient_modulation(hidden_states, is_augmented_tensor.any())

        # 获取全局特征（CLS token）
        global_features = hidden_states[:, 0, :]  # [batch_size * num_texts, hidden_size]

        # 应用投影层
        global_text_feats = self.pro_go(global_features)  # [batch_size * num_texts, embed_dim]
        local_text_feats = self.pro_lo(hidden_states)  # [batch_size * num_texts, seq_len, embed_dim]

        # 重塑维度以匹配输入格式
        global_text_feats = global_text_feats.view(batch_size, num_texts, self.embed_dim)
        local_text_feats = local_text_feats.view(batch_size, num_texts, seq_len, self.embed_dim)

        return global_text_feats, local_text_feats
