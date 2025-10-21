# batch_evaluate.py
# 여러 모델을 한 번에 평가하는 배치 스크립트

from evaluate_model import evaluate_model  # 위에서 만든 평가 함수 가져오기

# 평가할 모델들의 리스트
# 각 항목은 (모델경로, 로딩설정, 표시이름) 튜플
models = [
    # 첫 번째 모델: 한국어 Llama 8B (8비트 양자화)
    ("beomi/Llama-3-Open-Ko-8B", 
     "pretrained=beomi/Llama-3-Open-Ko-8B,load_in_8bit=True", 
     "Llama-3-Ko-8B"),
    
    # 두 번째 모델: Llama 3.2 3B (float16 정밀도)
    ("meta-llama/Llama-3.2-3B-Instruct", 
     "pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype=float16", 
     "Llama-3.2-3B"),
    
    # 여기에 추가 모델들을 계속 추가할 수 있음...
]

# 모델 리스트를 순회하며 하나씩 평가
for model_name, model_args, label in models:
    try:
        # 모델 평가 실행
        evaluate_model(model_name, model_args, label)
    except Exception as e:
        # 에러가 발생해도 다음 모델 평가를 계속 진행
        print(f"❌ Error with {label}: {e}")
        continue  # 다음 모델로 넘어감