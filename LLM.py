from openai import OpenAI
import os
import requests
import json
import time
class LLMClient:
    """封装LLM调用的客户端类"""
    
    def __init__(self, api_key=None, base_url=None, model_name="deepseek-r1-distill-llama-8b"):
        """
        初始化LLM客户端
        
        参数:
            api_key: API密钥，默认使用环境变量
            base_url: API基础URL
            model_name: 使用的模型名称
        """
        # 使用参数提供的key或默认值
        self.api_key = api_key or "5V-4NVFbjVkrzCEICdbkxNVL8CvwT1ghO56DLeEG4mMojPJswvKYVODa-o6oGbd-R6WiNOP14NiL6_VSixfSRA"
        self.base_url = base_url or "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/271c9332-4aa6-4ff5-95b3-0cf8bd94c394/v1"
        self.model_name = model_name
        
        # 初始化OpenAI客户端（保留以兼容原有代码）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def ask(self, question, stream=False, print_process=False):
        """
        向LLM提问并获取回答
        
        参数:
            question: 用户问题
            stream: 是否使用流式输出
            print_process: 是否打印思考过程和回复过程
            
        返回:
            dict: 包含思考过程和回答的字典
        """
        try:
            # 计算请求时间
            start_time = time.time()
            
            # 分割问题为系统提示和用户输入
            lines = question.split("\n")
            system_prompt = lines[0] if lines else "You are a helpful assistant"
            user_content = "\n".join(lines[1:]) if len(lines) > 1 else question
            
            # 使用OpenAI客户端发送请求
            response = self.client.chat.completions.create(
                model="DeepSeek-V3",  # 使用指定的模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.6,
                stream=stream
            )
            
            # 处理流式响应
            if stream:
                answer_content = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        answer_content += content_chunk
                        if print_process:
                            print(content_chunk, end="", flush=True)
            else:
                # 处理非流式响应
                answer_content = response.choices[0].message.content
            
            if print_process and not stream:
                print("\n" + "=" * 20 + "API返回内容" + "=" * 20 + "\n")
                print(answer_content)
                
            end_time = time.time()
            print(f"API请求时间: {end_time - start_time}秒")
            
            return {
                "reasoning": "",  # 当前版本不支持思考过程
                "answer": answer_content
            }
            
        except Exception as e:
            print(f"API调用出错: {e}")
            return {
                "reasoning": "",
                "answer": f"Error: {str(e)}"
            }
    
    def change_model(self, model_name):
        """更改使用的模型"""
        self.model_name = model_name
        return self
    
    def change_api_key(self, api_key):
        """更改API密钥"""
        self.api_key = api_key
        return self


# 示例用法
if __name__ == "__main__":
    llm = LLMClient()
    result = llm.ask("""
    作为新闻视觉分析师，需从新闻内容中提取以下7大要素，以严格JSON格式输出，确保每个要素为列表类型：
    1. 核心事件（名词短语，与标题紧密相关，1-3个）
    2. 关键动作（动宾结构，2-3个）
    3. 显著实体（具体名称，直接提取人名/物品名/机构名）
    4. 场景特征（环境/时间/视觉元素）
    5. 情感倾向（丰富描述）
    6. 视觉隐喻（象征性元素）
    7. 数据统计信息（新闻中的数据和统计信息）
    
    标题：大批香港市民在維多利亞公園欣賞彩燈。
    正文：9月15日，大批香港市民在維多利亞公園欣賞彩燈。今年維園的綵燈以"國風・港味"為主題，涵蓋衣、食、住、行四大生活範疇，展示各式糅合傳統與現代設計的璀璨綵燈，象徵中華文明繁榮昌盛，歷久彌新，呈現中華文化的瑰麗風韻和香港的地道風貌。圖為以唐裝與旗袍造型設計的12米高"衣"燈組。（香港中通社記者  謝光磊 攝）  香港中通社圖片
    """, stream=False, print_process=True)
    print("\n最终结果：")
    print(result["answer"])