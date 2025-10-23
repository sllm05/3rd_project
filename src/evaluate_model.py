# evaluate_model.py
# 개별 모델의 KMMLU 성능을 평가하고 결과를 저장하는 스크립트
# WandB 로깅 및 45개 전체 과목 순위 저장 기능 포함

# === 필요한 라이브러리 임포트 ===
from lm_eval import simple_evaluate
import csv
import json
import os
from datetime import datetime
import torch
import gc
import re
import time

# WandB 임포트 (선택적)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. To enable wandb logging, run: pip install wandb")

# === 전역 상수 정의 ===
CSV_FILE = 'kmmlu_results.csv'
JSON_LEADERBOARD = 'kmmlu_leaderboard.json'

# === KMMLU 45개 과목의 정확한 대분류 매핑 ===
KMMLU_SUBJECT_MAPPING = {
    "STEM": [
        "math", "physics", "chemistry", "biology", "earth_science",
        "computer_science", "electrical_engineering", "mechanical_engineering",
        "chemical_engineering", "civil_engineering", "information_technology"
    ],
    "HUMSS": [
        "korean_history", "world_history", "geography", 
        "korean_language_and_literature", "philosophy",
        "political_science_and_sociology", "economics", "law",
        "educational_psychology", "social_welfare",
        "human_development_and_family_studies", "business_administration",
        "accounting", "marketing"
    ],
    "Applied Science": [
        "agricultural_sciences", "food_processing", "animal_sciences",
        "fashion", "aviation_engineering_and_maintenance", "health",
        "nursing", "medicine", "dentistry", "pharmacology",
        "korean_medicine", "construction"
    ],
    "Other": [
        "public_safety", "defense", "nondestructive_testing",
        "industrial_engineer", "taxation", "labor_law",
        "patent", "real_estate"
    ]
}

# 역 매핑: 과목명 → 대분류
SUBJECT_TO_CATEGORY = {}
for category, subjects in KMMLU_SUBJECT_MAPPING.items():
    for subject in subjects:
        SUBJECT_TO_CATEGORY[subject] = category

def parse_model_config(model_args):
    """
    model_args 문자열에서 비트 정밀도 추출
    
    Args:
        model_args: "pretrained=...,load_in_8bit=True,dtype=float16" 형식의 문자열
    
    Returns:
        str: 비트 정밀도 (예: '8bit', 'float16', '4bit')
    """
    # load_in_8bit=True 체크
    if 'load_in_8bit=True' in model_args:
        return '8bit'
    # load_in_4bit=True 체크
    elif 'load_in_4bit=True' in model_args:
        return '4bit'
    # dtype 파라미터 체크
    elif 'dtype=' in model_args:
        dtype_match = re.search(r'dtype=(\w+)', model_args)
        if dtype_match:
            return dtype_match.group(1)
    
    return 'unknown'

def format_time(seconds):
    """
    초 단위 시간을 읽기 쉬운 형식으로 변환
    
    Args:
        seconds: 초 단위 시간
    
    Returns:
        str: 형식화된 시간 문자열 (예: "1h 23m 45s", "45m 30s", "30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def get_subject_category(subject_name, results=None):
    """
    개별 과목이 어떤 대분류에 속하는지 정확하게 판단
    """
    normalized = subject_name.replace('kmmlu_', '').lower()
    category = SUBJECT_TO_CATEGORY.get(normalized)
    
    if category:
        return category
    
    for cat, subjects in KMMLU_SUBJECT_MAPPING.items():
        for subj in subjects:
            if subj.replace('_', ' ') in normalized.replace('_', ' '):
                return cat
            if normalized.replace('_', ' ') in subj.replace('_', ' '):
                return cat
    
    print(f"⚠️ Warning: Unknown subject '{subject_name}', defaulting to 'Other'")
    return 'Other'

def load_results():
    """저장된 평가 결과 불러오기 (CSV에서)"""
    results = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('Model') or row.get('Model').startswith('---'):
                    continue
                results.append(row)
    return results

def save_results(results):
    """평가 결과를 CSV 파일로 저장"""
    save_to_csv(results)

def save_to_csv(results):
    """
    평가 결과를 CSV 파일로 저장
    구조: [기본 정보 섹션] + [전체 과목 순위 섹션]
    """
    if not results:
        return
    
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # === 섹션 1: 기본 모델 정보 ===
        writer.writerow(['=== BASIC MODEL INFORMATION ==='])
        writer.writerow([])
        
        # 헤더: Timestamp 제거, Elapsed_Time, Batch_Size, Precision 추가
        basic_header = ['Model', 'Overall', 'STEM', 'HUMSS', 'Applied', 'Other', 
                       'Best_Subject', 'Best_Score', 
                       'Worst_Subject', 'Worst_Score',
                       'Elapsed_Time', 'Batch_Size', 'Precision']
        writer.writerow(basic_header)
        
        # 각 모델의 기본 정보
        for result in results:
            basic_row = [
                result.get('model', result.get('Model', '')),
                f"{result['overall']:.4f}",
                f"{result['stem']:.4f}",
                f"{result['humss']:.4f}",
                f"{result['applied']:.4f}",
                f"{result['other']:.4f}",
                result['best']['name'],
                f"{result['best']['score']:.4f}",
                result['worst']['name'],
                f"{result['worst']['score']:.4f}",
                result.get('elapsed_time', 'N/A'),  # 걸린 시간
                result.get('batch_size', 'N/A'),    # 배치 크기
                result.get('precision', 'N/A')      # 비트 정밀도
            ]
            writer.writerow(basic_row)
        
        # === 섹션 2: 전체 과목 순위 (45개) ===
        writer.writerow([])
        writer.writerow(['=== ALL SUBJECTS RANKING ==='])
        writer.writerow([])
        
        for result in results:
            model_name = result.get('model', result.get('Model', ''))
            
            writer.writerow([f'Model: {model_name}'])
            writer.writerow(['Rank', 'Subject', 'Score', 'Category'])
            
            all_subjects = result.get('all_subjects_ranked', [])
            
            for rank, subject_info in enumerate(all_subjects, 1):
                writer.writerow([
                    rank,
                    subject_info['name'],
                    f"{subject_info['score']:.4f}",
                    subject_info['category']
                ])
            
            writer.writerow([])

def append_to_leaderboard(result_data):
    """
    평가 결과를 JSON 리더보드에 누적 저장
    
    Args:
        result_data: 평가 결과 딕셔너리
    """
    # 기존 리더보드 불러오기
    leaderboard = []
    if os.path.exists(JSON_LEADERBOARD):
        try:
            with open(JSON_LEADERBOARD, 'r', encoding='utf-8') as f:
                leaderboard = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            leaderboard = []
    
    # 같은 모델이 이미 있는지 확인
    existing_idx = None
    for i, entry in enumerate(leaderboard):
        if entry.get('model') == result_data['model'] or entry.get('model_path') == result_data['model_path']:
            existing_idx = i
            break
    
    # 기존 모델 있으면 업데이트, 없으면 추가
    if existing_idx is not None:
        leaderboard[existing_idx] = result_data
        print(f"📝 Updated existing entry in leaderboard: {result_data['model']}")
    else:
        leaderboard.append(result_data)
        print(f"➕ Added new entry to leaderboard: {result_data['model']}")
    
    # JSON 파일에 저장
    with open(JSON_LEADERBOARD, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Leaderboard saved to {JSON_LEADERBOARD}")

def evaluate_model(model_name, model_args, label, batch_size=16, wandb_project=None, wandb_run_name=None):
    """
    모델 평가 및 저장
    """
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # 비트 정밀도 추출
    precision = parse_model_config(model_args)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Batch Size: {batch_size}")
    print(f"Precision: {precision}")
    print(f"{'='*60}")
    
    # WandB 초기화
    wandb_run = None
    if WANDB_AVAILABLE and wandb_project:
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or label,
                config={
                    "model_name": model_name,
                    "model_args": model_args,
                    "label": label,
                    "batch_size": batch_size,
                    "precision": precision
                },
                reinit=True
            )
            print(f"✅ WandB logging enabled: {wandb_project}/{wandb_run_name or label}")
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            wandb_run = None
    
    # CUDA 메모리 최적화
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 평가 실행
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["kmmlu"],
        batch_size=batch_size,
        device="cuda:0",
        num_fewshot=5
    )
    
    # 종료 시간 기록 및 경과 시간 계산
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_time_str = format_time(elapsed_seconds)
    
    # 전체 평균 정확도
    overall = results['results']['kmmlu']['acc,none']
    
    # 대분류 점수
    stem_score = results['results']['kmmlu_stem']['acc,none']
    humss_score = results['results']['kmmlu_humss']['acc,none']
    applied_score = results['results']['kmmlu_applied_science']['acc,none']
    other_score = results['results']['kmmlu_other']['acc,none']
    
    # 개별 과목 점수 수집
    all_subjects = {}
    for task, metrics in results['results'].items():
        if task.startswith('kmmlu_') and 'acc,none' in metrics:
            if task not in ['kmmlu_stem', 'kmmlu_humss', 'kmmlu_applied_science', 'kmmlu_other', 'kmmlu']:
                subject_name = task.replace('kmmlu_', '').replace('_', ' ').title()
                category = get_subject_category(task)
                all_subjects[subject_name] = {
                    'score': metrics['acc,none'],
                    'category': category
                }
    
    # 점수 순으로 정렬
    sorted_subjects = sorted(all_subjects.items(), key=lambda x: x[1]['score'])
    
    all_subjects_ranked = [
        {
            "name": name,
            "score": info['score'],
            "category": info['category']
        }
        for name, info in sorted_subjects
    ]
    
    best_subject = sorted_subjects[-1]
    worst_subject = sorted_subjects[0]
    
    # 결과 정리
    result_data = {
        "model": label,
        "model_path": model_name,
        "overall": overall,
        "stem": stem_score,
        "humss": humss_score,
        "applied": applied_score,
        "other": other_score,
        "best": {
            "name": best_subject[0],
            "score": best_subject[1]['score']
        },
        "worst": {
            "name": worst_subject[0],
            "score": worst_subject[1]['score']
        },
        "all_subjects_ranked": all_subjects_ranked,
        "elapsed_time": elapsed_time_str,  # 걸린 시간 추가
        "batch_size": batch_size,          # 배치 크기 추가
        "precision": precision              # 비트 정밀도 추가
    }
    
    # WandB 로깅
    if wandb_run:
        try:
            wandb.log({
                "overall_accuracy": overall,
                "stem_accuracy": stem_score,
                "humss_accuracy": humss_score,
                "applied_science_accuracy": applied_score,
                "other_accuracy": other_score,
                "best_subject_score": best_subject[1]['score'],
                "worst_subject_score": worst_subject[1]['score'],
                "elapsed_seconds": elapsed_seconds,
                "batch_size": batch_size,
                "precision": precision
            })
            
            top_5 = all_subjects_ranked[-5:][::-1]
            bottom_5 = all_subjects_ranked[:5]
            
            wandb.log({
                "top_5_subjects": wandb.Table(
                    columns=["Subject", "Score", "Category"],
                    data=[[s['name'], s['score'], s['category']] for s in top_5]
                ),
                "bottom_5_subjects": wandb.Table(
                    columns=["Subject", "Score", "Category"],
                    data=[[s['name'], s['score'], s['category']] for s in bottom_5]
                )
            })
            
            print("✅ Results logged to WandB")
        except Exception as e:
            print(f"⚠️ WandB logging failed: {e}")
        finally:
            wandb.finish()
    
    # 콘솔 출력
    print(f"\n✅ Evaluation Complete!")
    print(f"Elapsed Time: {elapsed_time_str}")
    print(f"Overall: {overall:.2%}")
    
    print(f"\n📊 Category Scores:")
    print(f"  STEM:            {stem_score:.2%}")
    print(f"  HUMSS:           {humss_score:.2%}")
    print(f"  Applied Science: {applied_score:.2%}")
    print(f"  Other:           {other_score:.2%}")
    
    print(f"\n🏆 Best Subject:  {best_subject[0]:35s} {best_subject[1]['score']:.2%} ({best_subject[1]['category']})")
    print(f"📉 Worst Subject: {worst_subject[0]:35s} {worst_subject[1]['score']:.2%} ({worst_subject[1]['category']})")
    
    print(f"\n📋 Bottom 10 Subjects (낮은 점수):")
    for i, subject in enumerate(all_subjects_ranked[:10], 1):
        print(f"  {i:2d}. {subject['name']:35s} {subject['score']:.2%}  [{subject['category']}]")
    
    print(f"\nTotal Subjects Evaluated: {len(all_subjects)} (KMMLU has 45 standard subjects)")
    
    # 저장
    all_results = load_results()
    
    existing_idx = None
    for i, r in enumerate(all_results):
        if r.get('Model') == label or r.get('model_path') == model_name:
            existing_idx = i
            break
    
    if existing_idx is not None:
        all_results[existing_idx] = result_data
    else:
        all_results.append(result_data)
    
    # 1. CSV 저장
    save_results(all_results)
    print(f"\n💾 Saved to {CSV_FILE}\n")
    # 2. JSON에 누적 저장
    append_to_leaderboard(result_data)
    
    return result_data

if __name__ == "__main__":
    evaluate_model(
        model_name="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
        model_args="pretrained=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B,dtype=float16", # dtype=float16, load_in_8bit=True
        label="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
        batch_size=64
    )