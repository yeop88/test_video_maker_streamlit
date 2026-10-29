"""나레이션용 숫자 처리.

- read_numbers: "15번씩 3세트, 30초" → "열다섯 번씩 세 세트, 삼십 초".
  Chatterbox는 아라비아 숫자를 제대로 읽지 못하므로 음성 합성 직전에만 바꾼다 (자막은 숫자 그대로).
- number_values: 글에 나온 수량(숫자, "일곱 도" 같은 한글 수사, "eighteen" 같은 영어)을 값으로 모은다.
  LLM이 원본에 없는 숫자를 지어냈는지 확인할 때 쓴다.
"""
import re

SINO_DIGITS = "영일이삼사오육칠팔구"
SMALL_UNITS = ["", "십", "백", "천"]
LARGE_UNITS = ["", "만", "억", "조"]

NATIVE_ONES = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]

# 이 단위 앞의 1~99는 고유어로 읽는다 (예: 세 개, 열다섯 번)
NATIVE_COUNTERS = (
    "번", "개", "명", "마리", "살", "시간", "시", "세트", "잔", "권", "가지", "사람", "병", "그릇", "곡", "달", "군데",
)

NUMBER = re.compile(r"(\d[\d,]*)(\.\d+)?(\s?)([가-힣]*)")


def sino(n: int) -> str:
    if n == 0:
        return "영"
    groups = []
    while n:
        n, rest = divmod(n, 10000)
        groups.append(rest)
    words = []
    for i, group in reversed(list(enumerate(groups))):
        if group == 0:
            continue
        part = ""
        for pos in range(3, -1, -1):
            digit = group // 10**pos % 10
            if digit == 0:
                continue
            # 십·백·천 앞의 "일"은 읽지 않는다 (십오, 백, 천)
            part += ("" if digit == 1 and pos > 0 else SINO_DIGITS[digit]) + SMALL_UNITS[pos]
        if i == 1 and part == "일":  # 10000 → 만
            part = ""
        words.append(part + LARGE_UNITS[i])
    return "".join(words)


def native(n: int) -> str:
    if n == 20:
        return "스무"
    return NATIVE_TENS[n // 10] + NATIVE_ONES[n % 10]


def _replace(match: re.Match) -> str:
    integer, decimal, space, after = match.groups()
    n = int(integer.replace(",", ""))
    if not decimal and after.startswith("월") and n in (6, 10):  # 유월, 시월
        return ("유" if n == 6 else "시") + after
    if decimal:
        spoken = sino(n) + " 점 " + "".join(SINO_DIGITS[int(d)] for d in decimal[1:])
    elif 0 < n < 100 and after.startswith(NATIVE_COUNTERS):
        spoken = native(n)
    else:
        spoken = sino(n)
    # 단위와는 띄어 읽는다 (예: "삼십 초")
    return f"{spoken} {after}" if after else f"{spoken}{space}"


def read_numbers(text: str) -> str:
    return NUMBER.sub(_replace, text)


# 사실 확인이 필요한 수량 단위 ("번", "가지"처럼 말버릇으로도 쓰이는 단위는 뺀다)
FACT_UNITS = ("도", "시간", "시", "분", "초", "명", "개", "살", "년", "원", "달러", "퍼센트",
              "미터", "킬로미터", "킬로그램", "킬로", "그램", "층", "주", "일", "개월")
NATIVE_WORDS = {"하나": 1, "둘": 2, "셋": 3, "넷": 4, "스무": 20, **{w: i for i, w in enumerate(NATIVE_ONES) if w}}
TENS_WORDS = {w: (i * 10) for i, w in enumerate(NATIVE_TENS) if w}
ENGLISH_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
ENGLISH_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}


def parse_korean_number(word: str) -> int | None:
    """한글 수사를 값으로. 고유어("열여덟", "일곱")와 한자어("십팔", "이백")만 인정하고 아니면 None."""
    for tens, value in [("", 0), *TENS_WORDS.items()]:
        if tens and not word.startswith(tens):
            continue
        rest = word[len(tens):]
        if not rest and value:
            return value
        if rest in NATIVE_WORDS:
            return value + NATIVE_WORDS[rest]
    # 한자어는 "이", "오"처럼 한 글자짜리가 다른 낱말과 헷갈리므로 두 글자 이상만
    if len(word) < 2 or any(ch not in SINO_DIGITS + "십백천" for ch in word):
        return None
    total, digit = 0, 0
    for ch in word:
        if ch in SINO_DIGITS:
            digit = SINO_DIGITS.index(ch)
        else:
            total += (digit or 1) * 10 ** ("십백천".index(ch) + 1)
            digit = 0
    return total + digit


def number_values(text: str) -> set[int]:
    values = {int(d.replace(",", "")) for d in re.findall(r"\d[\d,]*", text)}
    # 어절마다: "일곱도", "다섯시에"처럼 수사+단위가 붙어 있거나, "다섯 시에"처럼 다음 어절이 단위로 시작
    words = re.findall(r"[가-힣]+", text)
    for i, word in enumerate(words):
        next_word = words[i + 1] if i + 1 < len(words) else ""
        candidates = [word] if next_word.startswith(FACT_UNITS) else []
        candidates += [word[:k] for k in range(1, len(word)) if word[k:].startswith(FACT_UNITS)]
        for candidate in candidates:
            value = parse_korean_number(candidate)
            if value is not None:
                values.add(value)
    lowered = text.lower()
    for i, word in enumerate(ENGLISH_ONES):
        if re.search(rf"\b{word}\b", lowered):
            values.add(i)
    for word, value in ENGLISH_TENS.items():
        for m in re.finditer(rf"\b{word}(?:[\s-]({'|'.join(ENGLISH_ONES[1:10])}))?\b", lowered):
            values.add(value + (ENGLISH_ONES.index(m.group(1)) if m.group(1) else 0))
    return values
