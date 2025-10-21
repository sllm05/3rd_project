# evaluate_model.py
# 개별 모델의 KMMLU 성능을 평가하고 결과를 저장하는 스크립트

# === 필요한 라이브러리 임포트 ===
from lm_eval import simple_evaluate  # lm-evaluation-harness 라이브러리의 핵심 평가 함수
import json  # JSON 파일 읽기/쓰기를 위한 표준 라이브러리
import os  # 파일 시스템 작업 (파일 존재 여부 확인 등)
from datetime import datetime  # 평가 시간을 기록하기 위한 날짜/시간 모듈
import wandb  # Weights & Biases - 머신러닝 실험을 추적하고 시각화하는 도구

# === 전역 상수 정의 ===
RESULTS_FILE = 'kmmlu_results.json'  # 모든 평가 결과를 저장할 JSON 파일 경로

def load_results():
    """저장된 평가 결과 불러오기"""
    # os.path.exists(): 파일이 실제로 존재하는지 확인
    if os.path.exists(RESULTS_FILE):
        # 파일이 있으면 읽기 모드('r')로 열기
        with open(RESULTS_FILE, 'r') as f:
            # JSON 텍스트를 파이썬 객체(리스트/딕셔너리)로 변환
            return json.load(f)
    # 파일이 없으면 빈 리스트 반환 (첫 실행시)
    return []

def save_results(results):
    """평가 결과를 JSON 파일로 저장"""
    # 쓰기 모드('w')로 파일 열기 (기존 내용은 덮어씀)
    with open(RESULTS_FILE, 'w') as f:
        # 파이썬 객체를 JSON 텍스트로 변환하여 파일에 저장
        # indent=2: 2칸 들여쓰기로 사람이 읽기 쉽게 포맷팅
        json.dump(results, f, indent=2)

def evaluate_model(model_name, model_args, label):
    """모델 평가 및 저장"""
    
    # === 1단계: 평가 시작 알림 ===
    print(f"\n{'='*60}")  # 60개의 '=' 문자로 구분선 출력
    print(f"Evaluating: {label}")  # 어떤 모델을 평가하는지 표시
    print(f"{'='*60}")
    
    # === 2단계: WandB 실험 추적 시작 ===
    # WandB는 실험의 하이퍼파라미터, 메트릭, 그래프 등을 자동으로 기록
    wandb.init(
        project="kmmlu-evaluation",  # WandB 대시보드에서 프로젝트 이름
        name=label,  # 이번 실험을 구분할 이름 (대시보드에 표시됨)
        config={"model": model_name}  # 실험 설정 메타데이터 기록
    )
    
    # === 3단계: 실제 모델 평가 실행 ===
    # simple_evaluate()는 lm-evaluation-harness의 핵심 함수
    # 모델을 로드하고, 데이터셋을 준비하고, 추론을 실행하여 점수를 계산
    results = simple_evaluate(
        model="hf",  # "hf"는 HuggingFace 모델을 사용한다는 의미
        model_args=model_args,  # 위에서 받은 모델 로딩 설정 전달
        tasks=["kmmlu"],  # 평가할 벤치마크 이름 (KMMLU는 한국어 MMLU)
        batch_size=8,  # 한 번에 몇 개의 샘플을 처리할지 (메모리에 따라 조절)
        device="cuda:0"  # 사용할 GPU 장치 (cuda:0은 첫 번째 GPU)
    )
    
    # === 4단계: 결과 데이터 추출 ===
    # results는 복잡한 중첩 딕셔너리 구조로 반환됨
    # results['results']['kmmlu']['acc,none']에서 전체 평균 정확도를 가져옴
    # 'acc,none'은 정규화하지 않은 원래 정확도를 의미
    overall = results['results']['kmmlu']['acc,none']
    
    # === 5단계: 개별 과목별 점수 수집 ===
    categories = {}  # 과목명: 점수를 저장할 딕셔너리
    
    # results['results']의 모든 항목을 순회
    for task, metrics in results['results'].items():
        # task 예시: 'kmmlu_biology', 'kmmlu_chemistry', 'kmmlu_stem' 등
        
        # 조건 1: kmmlu_로 시작하는 과목만 선택
        # 조건 2: 'acc,none' 메트릭이 있는지 확인
        if task.startswith('kmmlu_') and 'acc,none' in metrics:
            # 대분류(stem, humss, applied_science, other)는 제외
            # 우리는 개별 과목(biology, chemistry 등)만 필요함
            if task not in ['kmmlu_stem', 'kmmlu_humss', 'kmmlu_applied_science', 'kmmlu_other']:
                # 과목명 정리 작업
                # 1. 'kmmlu_' 접두사 제거
                # 2. '_'를 공백으로 변경
                # 3. 각 단어의 첫 글자를 대문자로 (Title Case)
                # 예: 'kmmlu_computer_science' → 'Computer Science'
                name = task.replace('kmmlu_', '').replace('_', ' ').title()
                categories[name] = metrics['acc,none']
    
    # 과목들을 점수 기준으로 내림차순 정렬 (높은 점수부터)
    # key=lambda x: x[1]은 튜플의 두 번째 요소(점수)를 기준으로 정렬
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    
    # === 6단계: 결과 데이터 구조화 ===
    # JSON 파일에 저장할 딕셔너리 생성
    result_data = {
        "model": label,  # 사용자가 지정한 표시용 이름
        "model_path": model_name,  # 실제 HuggingFace 모델 경로 (중복 확인용)
        "overall": overall,  # 전체 과목 평균 정확도
        
        # 4개 대분류 카테고리의 평균 점수
        "stem": results['results']['kmmlu_stem']['acc,none'],  # 과학/기술/공학/수학
        "humss": results['results']['kmmlu_humss']['acc,none'],  # 인문/사회과학
        "applied": results['results']['kmmlu_applied_science']['acc,none'],  # 응용과학
        "other": results['results']['kmmlu_other']['acc,none'],  # 기타 분야
        
        # 가장 잘한 과목 (정렬된 리스트의 첫 번째 항목)
        "best": {
            "name": sorted_cats[0][0],  # 과목명
            "score": sorted_cats[0][1]  # 점수
        },
        
        # 가장 못한 과목 (정렬된 리스트의 마지막 항목)
        "worst": {
            "name": sorted_cats[-1][0],  # 과목명
            "score": sorted_cats[-1][1]  # 점수
        },
        
        # 평가가 수행된 정확한 시간 기록 (ISO 8601 포맷)
        # 예: "2025-10-21T14:30:45.123456"
        "timestamp": datetime.now().isoformat()
    }
    
    # === 7단계: WandB에 핵심 메트릭 로깅 ===
    # 이 데이터들은 WandB 웹 대시보드에서 그래프로 시각화됨
    wandb.log({
        "overall": overall,  # 전체 점수
        "stem": result_data['stem'],  # STEM 점수
        "humss": result_data['humss'],  # 인문사회 점수
        "applied": result_data['applied'],  # 응용과학 점수
        "other": result_data['other']  # 기타 점수
    })
    
    # WandB 세션 종료 (리소스 정리)
    wandb.finish()
    
    # === 8단계: 콘솔에 결과 요약 출력 ===
    print(f"\n✅ Evaluation Complete!")
    # :.2% 포맷은 소수를 백분율로 표시 (소수점 2자리)
    # 예: 0.7523 → 75.23%
    print(f"Overall: {overall:.2%}")
    print(f"Best:    {result_data['best']['name']} ({result_data['best']['score']:.2%})")
    print(f"Worst:   {result_data['worst']['name']} ({result_data['worst']['score']:.2%})")
    
    # === 9단계: 결과를 JSON 파일에 저장 ===
    # 먼저 기존에 저장된 모든 결과 불러오기
    all_results = load_results()
    
    # 같은 모델이 이미 평가되었는지 확인
    # next()는 조건을 만족하는 첫 번째 인덱스를 반환, 없으면 None
    # enumerate()는 (인덱스, 값) 튜플을 생성
    idx = next((i for i, r in enumerate(all_results) if r['model_path'] == model_name), None)
    
    if idx is not None:
        # 기존 결과가 있으면 업데이트 (재평가한 경우)
        all_results[idx] = result_data
    else:
        # 기존 결과가 없으면 리스트에 새로 추가
        all_results.append(result_data)
    
    # 수정된 전체 결과를 파일에 저장
    save_results(all_results)
    print(f"Saved to {RESULTS_FILE}\n")
    
    # 평가 결과 딕셔너리 반환 (호출한 곳에서 사용 가능)
    return result_data

# === 메인 실행 블록 ===
# 이 파일이 직접 실행될 때만 아래 코드가 실행됨
# 다른 파일에서 import 할 때는 실행되지 않음
if __name__ == "__main__":
    # 평가하고 싶은 모델 설정 (여기만 수정하면 됨)
    evaluate_model(
        model_name="yanolja/YanoljaNEXT-EEVE-10.8B",  # HuggingFace에서 모델 다운로드
        model_args="pretrained=yanolja/YanoljaNEXT-EEVE-10.8B,load_in_8bit=True",  # 8비트 양자화로 메모리 절약 # load_in_8bit=True, dtype=float16
        label="YanoljaNEXT-EEVE-10.8B (8bit)"  # 결과 파일에 표시될 이름
    )