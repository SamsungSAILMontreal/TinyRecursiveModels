# utils/trajectory_utils.py

import torch
# IGNORE_LABEL_ID를 직접 임포트하여 코드의 명확성과 일관성을 보장합니다.
# 이는 models/losses.py에 정의된 상수로, 패딩된 레이블을 식별하는 데 사용됩니다.
from models.losses import IGNORE_LABEL_ID

class TrajectoryLog:
    """
    단일 데이터 샘플에 대한 '생각 궤적'과 관련 성능 지표를 저장하는 클래스.
    """
    def __init__(self, puzzle_id: str, aug_id: int, vectors: torch.Tensor):
        """
        Args:
            puzzle_id (str): 원본 퍼즐의 고유 ID.
            aug_id (int): 데이터 증강 샘플의 인덱스.
            vectors (torch.Tensor): [num_steps, hidden_dim] 형태의 궤적 벡터 시퀀스.
        """
        self.puzzle_id = puzzle_id
        self.aug_id = aug_id
        self.vectors = vectors
        
        # calculate_scores 메소드를 통해 채워질 값들
        self.final_accuracy: float = 0.0
        self.is_correct: bool = False

    def calculate_scores(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """
        모델의 최종 예측과 실제 정답을 비교하여 픽셀 단위 정확도와
        완전 정답 여부를 계산하고 인스턴스 변수에 저장합니다.

        이 로직은 프로젝트의 핵심 손실 함수인 `models.losses.CrossEntropyLoss_Ponder_Tokens`의
        정확도 계산 방식을 그대로 따릅니다.

        Args:
            prediction (torch.Tensor): 모델의 최종 예측 그리드 (토큰 ID 형태).
            ground_truth (torch.Tensor): 실제 정답 그리드.
        """
        # 1. 패딩/무시 레이블을 제외하기 위한 마스크 생성
        # ground_truth 텐서에서 IGNORE_LABEL_ID (-1) 값을 갖는 위치를 식별합니다.
        mask = ground_truth != IGNORE_LABEL_ID

        # 2. 유효한 픽셀의 총 개수 계산
        total_valid_pixels = mask.sum().item()

        # 유효한 픽셀이 없는 경우 (예: 전체가 패딩인 경우) 계산을 중단합니다.
        if total_valid_pixels == 0:
            self.final_accuracy = 0.0
            # 정답과 비교할 픽셀이 없으므로, 정답 여부는 False로 간주합니다.
            self.is_correct = False 
            return

        # 3. 마스크를 적용하여 유효한 픽셀만 추출
        masked_prediction = prediction[mask]
        masked_ground_truth = ground_truth[mask]

        # 4. 픽셀 단위 정확도 (Pixel Accuracy) 계산
        # 유효한 픽셀 중 예측과 정답이 일치하는 픽셀의 수를 셉니다.
        correct_pixels = (masked_prediction == masked_ground_truth).sum().item()
        self.final_accuracy = correct_pixels / total_valid_pixels

        # 5. 완전 일치 여부 (Exact Match) 계산
        # 유효한 픽셀 전체가 완벽하게 일치하는지 확인합니다.
        # `models.losses.py`에서는 배치 단위로 계산하지만, 여기서는 단일 샘플이므로
        # `correct_pixels`와 `total_valid_pixels`를 비교하는 것으로 동일한 결과를 얻습니다.
        self.is_correct = (correct_pixels == total_valid_pixels)