import os

# 告诉 Python：不管系统开没开代理，这个程序都不用代理
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 或者直接清空代理环境变量
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForMaskedLM
from transformers import EsmForProteinFolding


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
    
class ESMFoldPredictor:
    def __init__(self, model_name="facebook/esmfold_v1"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_resources(self):
        print(f"正在尝试从镜像站加载 ESMFold...")
        try:
            # 增加 trust_remote_code=True，有时某些配置文件需要它
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = EsmForProteinFolding.from_pretrained(
                self.model_name, 
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.to(self.device).eval()
            if self.device.type == "cuda":
                self.model = self.model.half()
            print("✅ ESMFold 加载成功！")
        except Exception as e:
            print(f"❌ 自动下载失败: {e}")
            print("建议方案：请使用下方提供的脚本进行离线下载。")

    def predict_structure_stability(self, sequence: str):
        """
        预测 3D 结构并提取平均 pLDDT 分数
        pLDDT 是 AlphaFold2/ESMFold 衡量局部稳定性的核心指标
        """
        seq = sequence.upper().strip()
        # ESMFold 限制长度通常为 1024 左右
        if len(seq) > 1024:
            print(f"警告：序列过长 ({len(seq)})，ESMFold 可能会溢出显存。")
            
        inputs = self.tokenizer([seq], return_tensors="pt", add_special_tokens=False).to(self.device)
        
        if self.device.type == "cuda":
            inputs = {k: v for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # pLDDT 存储在每个残基的预测置信度中
        # 范围通常是 0-1 (transformers 实现) 或 0-100
        plddt_per_residue = outputs.plddt[0] 
        avg_plddt = plddt_per_residue.mean().item()
        
        return avg_plddt

def run_test():
    '''
    用于测试蛋白质热稳定性评分程序（含序列概率评分与结构置信度评分）
    '''
    # 1. 初始化原有的基于 PLL 的预测器
    manager = ESMResourceManager()
    if not manager.load_resources():
        return
    predictor = StabilityPredictor(manager)

    # 2. 初始化新增的基于结构预测的 ESMFold (如果显存足够)
    # 注意：ESMFold 650M 加上推理开销，建议显存 > 12GB
    try:
        fold_predictor = ESMFoldPredictor()
        fold_predictor.load_resources()
    except Exception as e:
        print(f"ESMFold 加载失败（显存可能不足）: {e}")
        fold_predictor = None

    # 3. 待测试序列集
    test_cases = {
        "Wild Type": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKRECMMENDPDWR",
        "Test Type": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFQDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKRECMMENDPDWR",
        "TP53": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
        "Taq DNA Polymerase": "MRGMLPLFEPKGRVLLVDGHHLAYRTFHALKGLTTSRGEPVQAVYGFAKSLLKALKEDGDAVIVVFDAKAPSFRHEAYGGYKAGRAPTPEDFPRQLALIKELVDLLGLARLEVPGYEADDVLASLAKKAEKEGYEVRILTADKDLYQLLSDRIHVLHPEGYLITPAWLWEKYGLRPDQWADYRALTGDESDNLPGVKGIGEKTARKLLEEWGSLEALLKNLDRLKPAIREKILAHMDDLKLSWDLAKVRTDLPLEVDFAKRREPDRERLRAFLERLEFGSLLHEFGLLESPKALEEAPWPPPEGAFVGFVLSRKEPMWADLLALAAARGGRVHRAPEPYKALRDLKEARGLLAKDLSVLALREGLGLPPGDDPMLLAYLLDPSNTTPEGVARRYGGEWTEEAGERAALSERLFANLWGRLEGEERLLWLYREVERPLSAVLAHMEATGVRLDVAYLRALSLEVAEEIARLEAEVFRLAGHPFNLNSRDQLERVLFDELGLPAIGKTEKTGKRSTSAAVLEALREAHPIVEKILQYRELTKLKSTYIDPLPDLIHPRTGRLHTRFNQTATATGRLSSSDPNLQNIPVRTPLGQRIRRAFVAEEGYWLLVLDYSQIELRVLAHLSGDENLIRVFQEGRDIHTETASWMFGVPREAVDPLMRRAAKTINFGVLYGMSAHRLSQELAIPYEEAQAFIERYFQSFPKVRRAWIETTLEEGRRRGYVETLFGRRRYVPDLEARVKSVREAAERMAFNMPVQGTAADLMKLAMVKLFPRLREMGARMLLQVHDELVLEAAEAVARLAKEVMEGVYPLAVPLEVEVGIGEDWLSAKEA"
    }

    print("\n--- 综合稳定性评估结果 ---")
    header = f"{'Sequence Type':<20} | {'PLL Score':<12} | {'Avg pLDDT':<12}"
    print(header)
    print("-" * len(header))
    
    for label, seq in test_cases.items():
        # 1. 概率评分
        pll_score = predictor.calculate_score(seq)
        
        # 2. 结构置信度评分
        plddt_score = 0.0
        if fold_predictor:
            plddt_score = fold_predictor.predict_structure_stability(seq)
        
        print(f"{label:<20} | {pll_score:>12.4f} | {plddt_score:>12.4f}")

if __name__ == "__main__":
    run_test()