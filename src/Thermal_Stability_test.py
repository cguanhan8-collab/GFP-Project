import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForMaskedLM

# 告诉 Python：不管系统开没开代理，这个程序都不用代理
os.environ['NO_PROXY'] = 'hf-mirror.com' 
# 或者直接清空代理环境变量
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

class ESMResourceManager:
    """负责模型加载与硬件状态监控"""
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_resources(self):
        print(f"正在加载模型: {self.model_name} ...")

        try:
            # 尝试完全离线加载
            print("尝试从本地缓存直接读取...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                local_files_only=True
            )
            self.model = EsmForMaskedLM.from_pretrained(
                self.model_name, 
                local_files_only=True
            )
        except Exception:
            # 如果离线加载失败，再尝试联网下载
            print("本地文件不全，尝试通过镜像站补全...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = EsmForMaskedLM.from_pretrained(self.model_name)

        self.model.to(self.device).eval()
        print("✅ 加载成功！")
        return True

class StabilityPredictor:
    """蛋白质热稳定性评分引擎"""
    def __init__(self, resource_manager):
        self.rm = resource_manager

    def calculate_score(self, sequence: str):
        """
        class StabilityPredictor中的函数calculate_score

        用于稳定性评分！！！

        基于 Pseudo-Log-Likelihood (PLL) 的稳定性评分
        原理：自然界稳定的蛋白质序列在预训练模型中具有更高的概率密度
        """
        seq = sequence.upper().strip()
        inputs = self.rm.tokenizer(seq, return_tensors="pt").to(self.rm.device)
        
        with torch.no_grad():
            outputs = self.rm.model(**inputs)
            logits = outputs.logits  # [1, L+2, Vocab]
            
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 提取序列对应的 token 概率
        target_ids = inputs["input_ids"][0, 1:-1].unsqueeze(1) 
        actual_log_probs = log_probs[0, 1:-1, :].gather(1, target_ids)
        
        # 返回平均对数似然作为稳定性得分
        return actual_log_probs.mean().item()

def run_test():
    '''
    用于测试蛋白质热稳定性评分程序
    '''
    # 1. 初始化资源
    manager = ESMResourceManager()
    if not manager.load_resources():
        return
    # 2. 初始化预测器
    predictor = StabilityPredictor(manager)

    # 3. 待测试序列集 (现在又 野生型 vs 突变型示例，可以增加)
    # 在后面遍历评分
    test_cases = {
        "Wild Type": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKRECMMENDPDWR",
        "Test Type": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFQDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKRECMMENDPDWR",
        "BeforeTopSeq":"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDASYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSHYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        "Random Poly-A": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    }

    print("\n--- 稳定性评分测试结果 ---")
    print(f"{'Sequence Type':<15} | {'Stability Score':<15}")
    print("-" * 35)
    
    for label, seq in test_cases.items():
        score = predictor.calculate_score(seq)
        print(f"{label:<15} | {score:>15.4f}")

if __name__ == "__main__":
    run_test()