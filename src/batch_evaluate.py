# batch_evaluate.py
# 여러 모델을 한 번에 평가하는 배치 스크립트
# WandB 프로젝트에 모든 모델 결과를 로깅

from evaluate_model import evaluate_model

# WandB 프로젝트 설정 (선택)
WANDB_PROJECT = "kmmlu-evaluation"  # None으로 설정하면 WandB 비활성화

# 평가할 모델들의 리스트
# 각 항목은 (모델경로, 로딩설정, 표시이름) 튜플
models = [
    # 첫 번째 모델: EXAONE Deep 7.8B (8비트 양자화)
    ("LGAI-EXAONE/EXAONE-Deep-7.8B", 
     "pretrained=LGAI-EXAONE/EXAONE-Deep-7.8B,load_in_8bit=True", 
     "EXAONE-Deep-7.8B"),
    
    # 두 번째 모델: DeepSeek R1 Qwen 8B (8비트 양자화)
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", 
     "pretrained=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,load_in_8bit=True", 
     "DeepSeek-R1-Qwen3-8B"),
    
    # 세 번째 모델: Midm 2.0 Mini Instruct (float16 정밀도)
    ("K-intelligence/Midm-2.0-Mini-Instruct", 
     "pretrained=K-intelligence/Midm-2.0-Mini-Instruct,dtype=float16", 
     "Midm-2.0-Mini-Instruct"),
    
    # 네 번째 모델: SOLAR 10.7B (8비트 양자화)
    ("upstage/SOLAR-10.7B-v1.0", 
     "pretrained=upstage/SOLAR-10.7B-v1.0,load_in_8bit=True", 
     "SOLAR-10.7B-v1.0")
    
    # 여기에 추가 모델들을 계속 추가할 수 있음...
]

# 모델 리스트를 순회하며 하나씩 평가
for model_name, model_args, label in models:
    try:
        # WandB 실행 이름은 라벨을 소문자+하이픈으로 변환
        wandb_run_name = label.lower().replace('/', '-').replace('.', '-')
        
        # 모델 평가 실행
        evaluate_model(
            model_name=model_name,
            model_args=model_args,
            label=label,
            wandb_project=WANDB_PROJECT,  # WandB 프로젝트 (None이면 비활성화)
            wandb_run_name=wandb_run_name
        )
        
        print(f"✅ Successfully evaluated: {label}\n")
        
    except Exception as e:
        # 에러가 발생해도 다음 모델 평가를 계속 진행
        print(f"❌ Error with {label}: {e}")
        continue  # 다음 모델로 넘어감

print("\n" + "="*60)
print("🎉 Batch evaluation complete!")
print(f"Results saved to kmmlu_results.csv")
if WANDB_PROJECT:
    print(f"WandB results: https://wandb.ai/<your-username>/{WANDB_PROJECT}")
print("="*60)