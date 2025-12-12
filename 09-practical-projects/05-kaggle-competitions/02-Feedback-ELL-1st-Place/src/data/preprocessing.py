"""
数据预处理模块

本模块提供文本预处理和数据集划分功能，包括：
- 文本编码规范化和特殊字符处理
- 特殊标记（如学校名、学生名等）的替换
- 多标签分层交叉验证
- 序列长度分析

主要功能：
1. resolve_encodings_and_normalize: 处理混合编码问题
2. preprocess_text: 完整的文本预处理流程
3. make_folds: 创建多标签分层K折交叉验证
4. get_max_len_from_df: 确定数据集的最大序列长度
"""

import codecs
import re
from typing import Tuple
from text_unidecode import unidecode
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    """
    UTF-8编码错误处理器

    将编码错误的部分转换为UTF-8字节序列。
    这个函数会被codecs.register_error注册为自定义错误处理器。

    Args:
        error: Unicode编码错误对象

    Returns:
        (bytes, int): UTF-8编码的字节和下一个处理位置
    """
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    """
    解码错误处理器

    尝试使用CP1252编码解码失败的部分。
    CP1252是Windows西欧字符编码，常见于英文文本。

    Args:
        error: Unicode解码错误对象

    Returns:
        (str, int): 解码后的字符串和下一个处理位置
    """
    return error.object[error.start : error.end].decode("cp1252"), error.end


# 注册自定义编码错误处理器
# 这些处理器在后续的编码转换中被调用
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """
    解决文本编码问题并规范化

    处理包含多种编码的文本（如UTF-8和CP1252混合）。
    这个函数通过多次编码转换来修复编码问题，然后使用unidecode
    将所有Unicode字符转换为ASCII近似表示。

    处理流程：
    1. raw_unicode_escape编码 -> UTF-8解码（使用CP1252处理错误）
    2. CP1252编码 -> UTF-8解码（使用CP1252处理错误）
    3. unidecode规范化为ASCII

    Args:
        text: 待处理的文本字符串

    Returns:
        规范化后的ASCII文本

    Examples:
        >>> text = "Café"  # 包含非ASCII字符
        >>> result = resolve_encodings_and_normalize(text)
        >>> print(result)  # "Cafe"
    """
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def get_additional_special_tokens():
    """
    获取特殊标记替换映射

    返回一个字典，用于将数据中的占位符（如学校名、学生名等）
    替换为标准的特殊标记。这些特殊标记会被添加到tokenizer的词表中。

    特殊标记包括：
    - 换行符 -> [BR]
    - 学校名 -> [GENERIC_SCHOOL] / [SCHOOL_NAME]
    - 学生名 -> [STUDENT_NAME]
    - 地点名 -> [GENERIC_CITY] / [LOCATION_NAME]
    - 其他实体名称 -> 各种命名实体标记

    Returns:
        特殊标记替换字典，键为原始文本，值为标准标记

    Notes:
        - 这些标记帮助模型学习实体的语义角色而不是具体内容
        - 修正了数据中的拼写错误（如'Genric_Name'）
    """
    special_tokens_replacement = {
        '\n': '[BR]',
        'Generic_School': '[GENERIC_SCHOOL]',
        'Generic_school': '[GENERIC_SCHOOL]',
        'SCHOOL_NAME': '[SCHOOL_NAME]',
        'STUDENT_NAME': '[STUDENT_NAME]',
        'Generic_Name': '[GENERIC_NAME]',
        'Genric_Name': '[GENERIC_NAME]',  # 修正拼写错误
        'Generic_City': '[GENERIC_CITY]',
        'LOCATION_NAME': '[LOCATION_NAME]',
        'HOTEL_NAME': '[HOTEL_NAME]',
        'LANGUAGE_NAME': '[LANGUAGE_NAME]',
        'PROPER_NAME': '[PROPER_NAME]',
        'OTHER_NAME': '[OTHER_NAME]',
        'PROEPR_NAME': '[PROPER_NAME]',  # 修正拼写错误
        'RESTAURANT_NAME': '[RESTAURANT_NAME]',
        'STORE_NAME': '[STORE_NAME]',
        'TEACHER_NAME': '[TEACHER_NAME]',
    }
    return special_tokens_replacement


def replace_special_tokens(text):
    """
    替换文本中的特殊标记

    将文本中的占位符替换为标准的特殊标记。

    Args:
        text: 待处理的文本字符串

    Returns:
        替换后的文本字符串
    """
    special_tokens_replacement = get_additional_special_tokens()
    for key, value in special_tokens_replacement.items():
        text = text.replace(key, value)
    return text


def pad_punctuation(text):
    """
    在标点符号周围添加空格

    将标点符号与单词分开，便于tokenization。
    同时将多个连续空格合并为单个空格。

    Args:
        text: 待处理的文本字符串

    Returns:
        处理后的文本字符串

    Examples:
        >>> pad_punctuation("Hello,world!")
        "Hello , world ! "

    Notes:
        该函数目前未在主流程中使用，保留用于实验
    """
    text = re.sub('([.,!?()-])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    return text


def preprocess_text(text):
    """
    完整的文本预处理流程

    执行以下步骤：
    1. 解决编码问题并规范化为ASCII
    2. 替换特殊标记

    Args:
        text: 原始文本字符串

    Returns:
        预处理后的文本字符串

    Examples:
        >>> text = "Student STUDENT_NAME from Generic_School..."
        >>> result = preprocess_text(text)
        # "Student [STUDENT_NAME] from [GENERIC_SCHOOL]..."
    """
    text = resolve_encodings_and_normalize(text)
    text = replace_special_tokens(text)
    return text


def make_folds(df, target_cols, n_splits, random_state):
    """
    创建多标签分层K折交叉验证

    使用MultilabelStratifiedKFold确保每个fold中目标变量的分布
    与整体数据集保持一致。这对于回归任务特别重要。

    Args:
        df: 包含数据的DataFrame
        target_cols: 目标列名列表
        n_splits: 折数
        random_state: 随机种子，用于可复现性

    Returns:
        添加了'fold'列的DataFrame，fold列值为0到n_splits-1

    Notes:
        - 使用iterstrat库实现多标签分层
        - 对于多目标回归任务，这确保了各个目标的分布均衡
        - shuffle=True确保数据被随机打乱

    Examples:
        >>> df = pd.DataFrame({'text': [...], 'score1': [...], 'score2': [...]})
        >>> df = make_folds(df, ['score1', 'score2'], n_splits=5, random_state=42)
        >>> print(df['fold'].value_counts())  # 每个fold的样本数大致相等
    """
    kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for n, (train_index, val_index) in enumerate(kfold.split(df, df[target_cols])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    return df


def get_max_len_from_df(df, tokenizer, n_special_tokens=3):
    """
    计算数据集中的最大tokenized序列长度

    遍历所有文本，找出tokenization后的最大长度。
    用于动态设置模型的max_length参数。

    Args:
        df: 包含文本的DataFrame
        tokenizer: HuggingFace tokenizer对象
        n_special_tokens: 额外的特殊标记数量（如[CLS], [SEP]等）

    Returns:
        最大序列长度（包含特殊标记）

    Notes:
        - 不包含特殊标记的tokenization: add_special_tokens=False
        - 最终长度加上n_special_tokens以容纳[CLS], [SEP]等标记
        - 使用fillna("")处理空值

    Examples:
        >>> max_len = get_max_len_from_df(train_df, tokenizer)
        >>> print(f"Maximum sequence length: {max_len}")
    """
    lengths = []
    tk0 = tqdm(df['full_text'].fillna("").values, total=len(df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_length = max(lengths) + n_special_tokens
    return max_length
