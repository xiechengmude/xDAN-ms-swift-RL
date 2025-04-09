"""
搜索专家智能体的搜索质量评估模块
用于多轮迭代搜索优化
"""
import re
from typing import List, Dict, Any, Tuple


class SearchQualityEvaluator:
    """搜索质量评估器"""
    
    def __init__(self):
        # 评估维度权重
        self.dimension_weights = {
            'relevance': 0.4,    # 相关性
            'completeness': 0.3, # 完整性
            'reliability': 0.2,  # 可靠性
            'depth': 0.1         # 深度
        }
    
    def __call__(self, search_contents: List[str], sub_goals: List[str]) -> List[float]:
        """评估搜索结果质量
        
        参数:
            search_contents: 搜索结果内容列表
            sub_goals: 对应的子目标列表
            
        返回:
            quality_scores: 质量评分列表(0-1)
        """
        scores = []
        for content, goal in zip(search_contents, sub_goals):
            # 提取搜索结果的各个部分
            search_plan = self._extract_section(content, 'search_plan')
            search_process = self._extract_section(content, 'search_process')
            search_result = self._extract_section(content, 'search_result')
            evaluation = self._extract_section(content, 'evaluation')
            knowledge_graph = self._extract_section(content, 'knowledge_graph')
            
            # 计算各维度评分
            relevance_score = self._evaluate_relevance(search_result, goal)
            completeness_score = self._evaluate_completeness(
                search_result, search_plan, goal)
            reliability_score = self._evaluate_reliability(
                search_result, search_process)
            depth_score = self._evaluate_depth(
                search_result, knowledge_graph)
            
            # 加权计算总分
            total_score = (
                self.dimension_weights['relevance'] * relevance_score +
                self.dimension_weights['completeness'] * completeness_score +
                self.dimension_weights['reliability'] * reliability_score +
                self.dimension_weights['depth'] * depth_score
            )
            
            scores.append(total_score)
        
        return scores
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """从内容中提取特定部分"""
        pattern = f"<{section_name}>(.*?)</{section_name}>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _evaluate_relevance(self, search_result: str, goal: str) -> float:
        """评估搜索结果与子目标的相关性"""
        # 简单实现：检查关键词匹配度
        # 实际应用中应使用更复杂的语义相似度计算
        if not search_result or not goal:
            return 0.0
            
        goal_keywords = self._extract_keywords(goal)
        result_text = search_result.lower()
        
        if not goal_keywords:
            return 0.5  # 无法提取关键词时给予中等分数
            
        matches = sum(1 for kw in goal_keywords if kw.lower() in result_text)
        return min(1.0, matches / max(1, len(goal_keywords)))
    
    def _evaluate_completeness(self, search_result: str, 
                              search_plan: str, goal: str) -> float:
        """评估搜索结果的完整性"""
        if not search_result:
            return 0.0
            
        # 检查是否覆盖了计划中的所有要点
        plan_points = self._extract_plan_points(search_plan)
        if not plan_points:
            # 如果没有明确的计划点，则基于内容长度和结构评估
            return min(1.0, len(search_result) / 500) * 0.7
            
        result_text = search_result.lower()
        covered_points = sum(1 for point in plan_points 
                           if point.lower() in result_text)
        
        return min(1.0, covered_points / max(1, len(plan_points)))
    
    def _evaluate_reliability(self, search_result: str, 
                             search_process: str) -> float:
        """评估搜索结果的可靠性"""
        if not search_result:
            return 0.0
            
        # 检查是否引用了来源
        source_indicators = ['根据', '来源', '引用', '参考', 
                            'according to', 'source', 'reference']
        has_sources = any(indicator in search_result.lower() 
                         for indicator in source_indicators)
        
        # 检查搜索过程是否详细
        process_detail = min(1.0, len(search_process) / 300) if search_process else 0.0
        
        # 检查是否包含数据或具体事实
        has_data = bool(re.search(r'\d+(\.\d+)?%|\d+年|\d{4}', search_result))
        
        return (0.4 * float(has_sources) + 
                0.4 * process_detail + 
                0.2 * float(has_data))
    
    def _evaluate_depth(self, search_result: str, 
                       knowledge_graph: str) -> float:
        """评估搜索深度"""
        if not search_result:
            return 0.0
            
        # 基于内容长度的基础分
        length_score = min(1.0, len(search_result) / 1000)
        
        # 知识图谱完整性
        kg_score = min(1.0, len(knowledge_graph) / 200) if knowledge_graph else 0.0
        
        # 检查是否包含深入分析
        analysis_indicators = ['分析', '评估', '比较', '优缺点', 
                              'analysis', 'evaluation', 'comparison']
        has_analysis = any(indicator in search_result.lower() 
                          for indicator in analysis_indicators)
        
        return (0.4 * length_score + 
                0.3 * kg_score + 
                0.3 * float(has_analysis))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单实现：分词并过滤停用词
        # 实际应用中应使用更复杂的关键词提取算法
        if not text:
            return []
            
        words = re.findall(r'\w+', text)
        stopwords = {'的', '了', '和', '与', '或', '在', '是', '有', 
                    'the', 'and', 'or', 'in', 'is', 'are', 'to', 'of'}
        
        return [w for w in words if len(w) > 1 and w.lower() not in stopwords]
    
    def _extract_plan_points(self, search_plan: str) -> List[str]:
        """从搜索计划中提取要点"""
        if not search_plan:
            return []
            
        # 尝试提取列表项
        list_items = re.findall(r'[•\-\d+]\s*(.*?)(?:\n|$)', search_plan)
        if list_items:
            return list_items
            
        # 如果没有明确的列表，尝试按句子分割
        sentences = re.split(r'[.。!！?？;；]+', search_plan)
        return [s.strip() for s in sentences if s.strip()]


def search_quality_evaluator(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """多轮搜索质量评估函数
    
    用于GRPO训练中的多轮对话控制
    评估搜索结果质量，并为质量不足的结果添加改进提示
    """
    evaluator = SearchQualityEvaluator()
    refine_prompt = "\n\n我需要改进我的搜索策略，因为当前结果存在以下不足："
    
    # 获取当前搜索结果和子目标
    search_contents = [input['messages'][-1]['content'] for input in inputs]
    sub_goals = [input.get('sub_goal', '全面搜索相关信息') for input in inputs]
    
    # 评估搜索质量
    quality_scores = evaluator(search_contents, sub_goals)
    
    # 处理每个搜索结果
    for score, input in zip(quality_scores, inputs):
        content = input['messages'][-1]['content']
        
        # 如果搜索质量不够好且尚未包含优化提示
        if score < 0.8 and refine_prompt not in content:
            # 确定需要改进的方面
            improvement_areas = []
            
            # 提取各部分内容
            search_result = evaluator._extract_section(content, 'search_result')
            search_process = evaluator._extract_section(content, 'search_process')
            knowledge_graph = evaluator._extract_section(content, 'knowledge_graph')
            evaluation = evaluator._extract_section(content, 'evaluation')
            
            # 评估各维度并确定改进方向
            relevance_score = evaluator._evaluate_relevance(
                search_result, input.get('sub_goal', ''))
            completeness_score = evaluator._evaluate_completeness(
                search_result, evaluator._extract_section(content, 'search_plan'), 
                input.get('sub_goal', ''))
            reliability_score = evaluator._evaluate_reliability(
                search_result, search_process)
            depth_score = evaluator._evaluate_depth(
                search_result, knowledge_graph)
            
            # 根据各维度评分确定改进方向
            if relevance_score < 0.7:
                improvement_areas.append("- 搜索结果与目标相关性不足，需要更精确地对准搜索目标")
            if completeness_score < 0.7:
                improvement_areas.append("- 搜索结果不够全面，未覆盖目标的所有关键方面")
            if reliability_score < 0.7:
                improvement_areas.append("- 搜索结果缺乏可靠来源或证据支持")
            if depth_score < 0.7:
                improvement_areas.append("- 搜索深度不足，需要提供更深入的分析和见解")
            
            # 如果没有具体改进方向，添加通用建议
            if not improvement_areas:
                improvement_areas.append("- 整体搜索质量有待提高，需要更全面、深入的信息收集")
            
            # 保留搜索过程，移除可能的结论部分
            if '<evaluation>' in content:
                content = content[:content.index('<evaluation>')]
            if '</search_result>' in content:
                content = content[:content.index('</search_result>') + len('</search_result>')]
                
            # 添加优化提示，引导智能体思考如何改进搜索策略
            content += refine_prompt + "\n" + "\n".join(improvement_areas)
            input['messages'][-1]['content'] = content
            input['finished'] = False  # 标记为未完成，需要继续优化
        else:
            input['finished'] = True  # 标记为已完成
            
    return inputs
