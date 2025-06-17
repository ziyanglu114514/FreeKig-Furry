import requests
import re
class LiuYingCosplayRewriter:
    def __init__(self, api_key=None, model="gpt-4o", base_url="https://api.openai.com/v1"):
        """
        :param api_key: 你的openai或本地服务的api key。若本地服务可不填。
        :param model:   openai的模型名（本地服务对应的模型名）
        :param base_url: openai或本地兼容API的base url（如 "http://localhost:8000/v1"）
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompt_template = '''
先验背景：
当前你正在协助一位用户在进行kigurumi角色扮演（cosplay流萤）。环境嘈杂，用户无法说长句，只能输入简短的文字。

由于现场噪音和识别误差，用户输入的短句可能不够完整或表达不清楚。你需要结合场景和上下文，合理推测用户的真实意图，并补充细节，使对话更自然贴合角色。

用户希望自己的每一句话都经过你润色，变成流萤风格的温柔少女语气台词（见下方范例）。

任务要求：
尽量还原并推理用户短句背后的真实意图，但不能脱离其原本大意。

结合流萤角色设定与场景补充细节、情感与适当环境描述，使台词更有“流萤”特色（可以适度延展，但不要离题）。

输出风格、口吻与措辞要与下方范例一致，语言柔和、富有少女气息、略带活泼或自嘲，能自然引导对话或增强场景感。

不可机械重复用户原句，不要引入用户意图外的无关剧情。

字数1–2句为宜，可补充情感和现场氛围，但不可冗长。

角色台词风格范例（仅供风格参考）：
- “你好呀，又见面啦…我的意思，很高兴见到你。和往常一样，叫我‘流萤’吧。”
- “下次见啦，今天的美好时光又要结束了。”
- “我…其实只是临时演员，如果你愿意，我可以带你体验格拉克斯大道上的各种有趣事物！”
- “这里是匹诺康尼，梦想汇聚的地方，盛大筵席永不落幕！”
- “唔，所以每天只能吃一个橡木蛋糕卷……没、没问题的！反正是在梦里……”

请将用户输入的短句【{user_input}】将其润色成贴合当前流萤角色风格的台词。如有必要，可结合环境和角色身份做合适补充。只输出台词，不要加任何解释。你只是在做文风转换，你没有在和用户对话。/no_think
        '''

    def replace_commas(self, text):
    # 使用全角逗号分割字符串
        segments = text.split('，')
        n = len(segments)
        if n <= 1:
            return text
        
        # 初始化一个列表，标记每个逗号位置是否需要替换为半角逗号
        replace_markers = [False] * (n - 1)
        
        # 处理每个逗号位置
        for i in range(n - 1):
            # 条件1：当前片段长度<=2，且选择与右侧合并（右侧片段更短或相等）
            if len(segments[i]) <= 2:
                if i == 0:  # 左侧无片段，只能向右合并
                    replace_markers[i] = True
                else:
                    # 比较左侧片段(i-1)和右侧片段(i+1)的长度
                    left_len = len(segments[i - 1])
                    right_len = len(segments[i + 1])
                    if right_len <= left_len:
                        replace_markers[i] = True
            
            # 条件2：下一个片段长度<=2，且选择与左侧合并（左侧片段更短或相等）
            if len(segments[i + 1]) <= 2:
                if i + 1 == n - 1:  # 右侧无片段，只能向左合并
                    replace_markers[i] = True
                else:
                    # 比较左侧片段(i)和右侧片段(i+2)的长度
                    left_len = len(segments[i])
                    right_len = len(segments[i + 2])
                    if left_len <= right_len:
                        replace_markers[i] = True
        
        # 重新构建结果字符串
        result = [segments[0]]
        for i in range(n - 1):
            # 根据标记决定使用半角逗号还是全角逗号
            if replace_markers[i]:
                result.append(',')
            else:
                result.append('，')
            result.append(segments[i + 1])
        
        return ''.join(result)

    def post_process(self, output):
        # 移除所有形如 <xxx>...</xxx> 的内容
        cleaned = re.sub(r'<[^<>]+>.*?</[^<>]+>', '', output, flags=re.DOTALL)
        # 只保留最后一段/句，如果有多余空行，只取最后一个非空行
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        return lines[-1] if lines else ""
    
    def rewrite(self, user_input):
        prompt = self.prompt_template.format(user_input=user_input)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:  # 本地服务如果不校验api_key可不填
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
    "model": self.model,
    "messages": [
        {
            "role": "system",
            "content":
                    """
                    你是台词风格润色助手。你的任务是：不与用户互动、不模拟对话，只把用户输入的简短内容，润色为温柔、活泼、少女感的“流萤”风格台词（不可直接照搬原句，也不要主动展开剧情）。只输出一到两句风格化台词，不输出解释或角色前缀。
                    """
                            },
                            {"role": "user", "content": user_input}
                        ],
                        "temperature": 0.4,
                        "max_tokens": 60
                    }
        response = requests.post(url, headers=headers, json=payload)
        if not response.ok:
            raise RuntimeError(f"API返回异常: {response.status_code} {response.text}")

        data = response.json()

        data=data["choices"][0]["message"]["content"].strip()
        data=self.post_process(data)
        data=self.replace_commas(data)
        return data


# 用法示例（OpenAI云端或本地都支持）
if __name__ == "__main__":
    # openai官方：api_key="sk-xxx", base_url="https://api.openai.com/v1", model="gpt-3.5-turbo"等
    # 本地llm（如llama.cpp server）：api_key=None, base_url="http://localhost:8000/v1", model="your-model-name"
    liuying = LiuYingCosplayRewriter(
        api_key=None,                        # 本地LLM可以不填api_key
        model="GPT-4.1",             # 或你实际的模型名
        # base_url="http://192.168.31.186:1234/v1"  # 本地OpenAI格式API地址
    )
    print(liuying.rewrite("你好"))