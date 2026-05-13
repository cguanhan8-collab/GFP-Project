import os
import pandas as pd
import re

class GFPDataLoader:
    def __init__(self, data_subdir='data'):
        self.data_dir = os.path.join(os.getcwd(), data_subdir)
        self.train_data_path = os.path.join(self.data_dir, 'GFP_data.xlsx')
        self.wt_seq_path = os.path.join(self.data_dir, 'AAseqs of 5 GFP proteins.txt')
        
        self.df_train = None
        self.wt_data = {}  # 存储结构: { 'sfGFP': {'seq': '...', 'pdb': '2B3P'}, ... }

    def load_all(self):
        """一键加载所有数据"""
        try:
            self._load_excel()
            self._parse_fasta_txt()
            print("数据全部加载成功！")
        except Exception as e:
            print(f"加载失败: {e}")

    def _load_excel(self):
        """读取训练 Excel"""
        if os.path.exists(self.train_data_path):
            self.df_train = pd.read_excel(self.train_data_path)
            print(f"统计: 已载入 {len(self.df_train)} 条实验数据")
        else:
            print(f"警告: 未找到 Excel 文件 {self.train_data_path}")

    def _parse_fasta_txt(self):
        """
        解析野生型文本文件
        规则：序列起始于 >ID 行，终止于 # 注释行。
        """
        if not os.path.exists(self.wt_seq_path):
            raise FileNotFoundError(f"找不到文件: {self.wt_seq_path}")

        # 清空旧数据，防止重复加载
        self.wt_data = {}

        with open(self.wt_seq_path, 'r', encoding='utf-8') as f:
            current_id = None
            can_collect_seq = False  # 序列收集开关

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 1. 匹配 ID 行 (>sfGFP)
                if line.startswith('>'):
                    current_id = line[1:].strip()
                    self.wt_data[current_id] = {'seq': '', 'pdb': None}
                    can_collect_seq = True  # 开启收集
                
                # 2. 匹配注释行 (# recommend PDB: 2B3P)
                elif line.startswith('#'):
                    can_collect_seq = False  # 关键：一旦遇到 #，立即停止收集序列内容
                    if current_id and 'PDB:' in line:
                        pdb_id = line.split('PDB:')[-1].strip()
                        self.wt_data[current_id]['pdb'] = pdb_id
                
                # 3. 匹配序列行
                elif current_id and can_collect_seq:
                    # 过滤掉可能的空格或数字，只保留氨基酸大写字母
                    clean_seq = re.sub(r'[^A-Z]', '', line.upper())
                    self.wt_data[current_id]['seq'] += clean_seq

        # --- 详细输出解析结果，用于校验 ---
        print("\n" + "="*70)
        print(f"{'GFP Type':<18} | {'Len':<5} | {'PDB':<8} | {'Sequence Preview'}")
        print("-" * 70)
        
        for gfp_id, info in self.wt_data.items():
            seq = info['seq']
            pdb = info['pdb'] if info['pdb'] else "N/A"
            length = len(seq)
            
            # 序列预览逻辑
            if length > 30:
                preview = f"{seq[:12]}...{seq[-12:]}"
            else:
                preview = seq
                
            print(f"{gfp_id:<18} | {length:<5} | {pdb:<8} | {preview}")
        
        print("-" * 70)
        print(f"统计: 已成功解析 {len(self.wt_data)} 个野生型蛋白序列")
        print("="*70 + "\n")

    def get_mutant_data(self):
        """
        专门为 MutantGenerator 提供的接口
        返回加载好的实验数据 DataFrame (包含 aaMutations, GFP type, Brightness 等)
        """
        if self.df_train is not None:
            return self.df_train
        else:
            print("❌ 错误: df_train 为空，请先运行 load_all()")
            return None
        
    # --- 外部访问接口 ---
    def get_wt_sequence(self, name):
        """获取指定 GFP 的纯序列"""
        return self.wt_data.get(name, {}).get('seq')

    def get_pdb_code(self, name):
        """获取指定 GFP 推荐的 PDB ID"""
        return self.wt_data.get(name, {}).get('pdb')

    def list_available_gfps(self):
        """查看目前加载了哪些蛋白"""
        return list(self.wt_data.keys())

class MutantGenerator:
    """
    专门负责将突变应用到野生型序列上的类。
    """
    def __init__(self, data_loader):
        """
        :param data_loader: 已经加载好数据的 GFPDataLoader 实例
        """
        self.loader = data_loader
        self.df_processed = None

    def _parse_single_mutation(self, base_seq_list, mut_code):
        """
        内部方法：解析单个突变（如 'A109D'）并修改氨基酸列表
        """
        # 使用正则表达式匹配：原氨基酸(A-Z)、位置(\d+)、新氨基酸(A-Z)
        match = re.match(r'([A-Z])(\d+)([A-Z])', mut_code)
        if not match:
            return False
            
        old_aa, pos, new_aa = match.groups()
        index = int(pos)  # 转换成 0 索引
        
        # 边界检查与原氨基酸校验
        if 0 <= index < len(base_seq_list):
            if base_seq_list[index] == old_aa:
                base_seq_list[index] = new_aa
                # print(f"✅")
                return True
            else:
                # 如果校验失败，可能是野生型序列匹配错了
                print(f"⚠️ 校验失败: 位置 {pos} 原本是 {base_seq_list[index]} 而非 {old_aa}")
        return False

    def generate_mutated_sequence(self, gfp_type, mutation_str):
        """
        根据 GFP 类型和突变字符串生成完整序列
        """
        # 1. 从 loader 获取对应的野生型序列
        raw_wt = self.loader.get_wt_sequence(gfp_type)
        if not raw_wt:
            return None
        
        # 转换为列表方便修改（字符串在 Python 中不可变）
        seq_list = list(raw_wt)
        
        # 2. 处理野生型情况
        if mutation_str == 'WT' or pd.isna(mutation_str):
            return "".join(seq_list)
        
        # 3. 处理多点突变 (用冒号分割)
        mutations = str(mutation_str).split(':')
        for mut in mutations:
            self._parse_single_mutation(seq_list, mut.strip())
            
        return "".join(seq_list)

    def _reverse_engineer_mutations(self, full_seq, gfp_type):
        """
        逆向推导：通过对比 Full Sequence 和 WT Sequence 找出突变点
        """
        wt_seq = self.loader.get_wt_sequence(gfp_type)
        ################## print(wt_seq)
        
        if not wt_seq or not full_seq:
            return "N/A"
        
        # 如果长度不一致，说明发生了插入或缺失（Indel），标记为错误
        if len(full_seq) != len(wt_seq):
            return "Error:LengthMismatch"

        diffs = []
        for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, full_seq)):
            if wt_aa != mut_aa:
                # 生物学编号从 1 开始
                diffs.append(f"{wt_aa}{i}{mut_aa}")
        
        return ":".join(diffs) if diffs else "WT"

    def process_dataframe(self, output_name=r'data/full_sequences.csv'):
        """
        处理数据，添加完整序列和逆向校验列，并保存为 CSV
        """
        df = self.loader.get_mutant_data()
        if df is None:
            print("❌ 错误: DataLoader 中没有可用的数据帧")
            return None
            
        print(f"正在处理 {len(df)} 条数据...")
        
        # 1. 复制数据
        self.df_processed = df.copy()
        
        # 2. 生成完整序列 (Full Sequence)
        self.df_processed['Full_Sequence'] = self.df_processed.apply(
            lambda row: self.generate_mutated_sequence(row['GFP type'], row['aaMutations']),
            axis=1
        )
        
        # 3. 逆向推导突变点 (Verification_Mutations)
        # 这一步通过比较 Full_Sequence 和 WT 来确保我们的 generate 逻辑没问题
        self.df_processed['Verification_Mutations'] = self.df_processed.apply(
            lambda row: self._reverse_engineer_mutations(row['Full_Sequence'], row['GFP type']),
            axis=1
        )
        
        # 4. 选择需要的列并保存
        columns_to_save = ['aaMutations', 'GFP type', 'Brightness', 'Full_Sequence', 'Verification_Mutations']
        
        # 确保只保存存在的列（防止 Excel 里 Brightness 拼写不一致）
        actual_columns = [col for col in columns_to_save if col in self.df_processed.columns]
        
        output_dir = os.path.dirname(output_name)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建文件夹: {output_dir}")

        try:
            self.df_processed[actual_columns].to_csv(output_name, index=False, encoding='utf-8-sig')
            print(f"✅ 文件已成功生成: {os.path.join(os.getcwd(), output_name)}")
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            
        return self.df_processed
    

# --- 测试运行 ---
if __name__ == "__main__":
    # loader = GFPDataLoader()
    # loader.load_all()

    # # 示例操作
    # target = 'avGFP'
    # if target in loader.list_available_gfps():
    #     print(f"\n--- {target} 信息 ---")
    #     print(f"PDB 编号: {loader.get_pdb_code(target)}")
    #     print(f"序列长度: {len(loader.get_wt_sequence(target))}")
    #     print(f"序列预览: {loader.get_wt_sequence(target)[:250]}")
    try:
        # 第一步：实例化加载器并读取文件
        my_loader = GFPDataLoader(data_subdir='data')
        my_loader.load_all()

        # 第二步：实例化处理器，并传入加载器
        my_processor = MutantGenerator(my_loader)
        my_processor.process_dataframe()
        
    except Exception as e:
        print(f"程序运行中出错: {e}")