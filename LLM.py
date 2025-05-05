from openai import OpenAI
import os

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
        self.api_key = api_key or "sk-bb4a4070a95b43aeab5aeb7c9095d2c1"
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = model_name
        
        # 初始化OpenAI客户端
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
        reasoning_content = ""  # 思考过程
        answer_content = ""     # 完整回复
        is_answering = False    # 是否开始回复
        
        # 创建聊天完成请求
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": question}
            ],
            stream=stream,
            # 解除以下注释会在最后一个chunk返回Token使用量
            # stream_options={
            #     "include_usage": True
            # }
        )
        
        if print_process:
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
        
        # 如果使用流式输出
        if stream:
            for chunk in completion:
                # 如果chunk.choices为空，则是usage信息
                if not chunk.choices:
                    if print_process and hasattr(chunk, 'usage'):
                        print("\nUsage:")
                        print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    # 处理思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        if print_process:
                            print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    # 处理回答内容
                    elif hasattr(delta, 'content') and delta.content is not None:
                        # 开始回复时的分隔
                        if delta.content != "" and is_answering == False:
                            is_answering = True
                            if print_process:
                                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        # 打印回复过程
                        if print_process:
                            print(delta.content, end='', flush=True)
                        answer_content += delta.content
        else:
            # 非流式输出处理
            response = completion
            if hasattr(response.choices[0], 'message'):
                answer_content = response.choices[0].message.content
            
        if print_process:
            print("\n" + "=" * 20 + "完整思考过程" + "=" * 20 + "\n")
            print(reasoning_content)
            print("=" * 20 + "完整回复" + "=" * 20 + "\n")
            print(answer_content)
            

        print({"reasoning": reasoning_content, "answer": answer_content})
        return {
            "reasoning": reasoning_content,
            "answer": answer_content
        }
    
    def change_model(self, model_name):
        """更改使用的模型"""
        self.model_name = model_name
        return self
    
    def change_api_key(self, api_key):
        """更改API密钥"""
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        return self


# 示例用法
if __name__ == "__main__":
    llm = LLMClient()
    result = llm.ask("9.9和9.11谁大", stream=False, print_process=False)
    print("final answer")
    print(result["answer"])
    # 可以直接使用result["answer"]和result["reasoning"]进行后续处理