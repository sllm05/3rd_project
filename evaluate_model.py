# evaluate_model.py
# 개별 모델의 KMMLU 성능을 평가하고 결과를 저장하는 스크립트

# === 필요한 라이브러리 임포트 ===
from lm_eval import simple_evaluate
import csv  # CSV 파일 저장용
import os
from datetime import datetime
import torch
import gc

# === 전역 상수 정의 ===
CSV_FILE = 'kmmlu_results.csv'  # CSV 파일

def load_results():
    """저장된 평가 결과 불러오기 (CSV에서)"""
    results = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    return results

def save_results(results):
    """평가 결과를 CSV 파일로 저장"""
    save_to_csv(results)

def save_to_csv(results):
    """평가 결과를 CSV 파일로 저장"""
    if not results:
        return
    
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더 작성
        header = ['Model', 'Overall', 'STEM', 'HUMSS', 'Applied', 'Other', 
                 'Best_Subject', 'Best_Score']
        # 최하위 10개 과목 헤더 추가
        for i in range(1, 11):
            header.extend([f'Worst_{i}_Subject', f'Worst_{i}_Score'])
        header.append('Timestamp')
        writer.writerow(header)
        
        # 데이터 작성
        for result in results:
            row = [
                result.get('model', result.get('Model', '')),
                f"{result['overall']:.4f}",
                f"{result['stem']:.4f}",
                f"{result['humss']:.4f}",
                f"{result['applied']:.4f}",
                f"{result['other']:.4f}",
                result['best']['name'],
                f"{result['best']['score']:.4f}",
            ]
            
            # 최하위 10개 과목 추가
            for worst in result.get('worst_10', []):
                row.extend([worst['name'], f"{worst['score']:.4f}"])
            
            # 10개 미만이면 빈 칸 채우기
            remaining = 10 - len(result.get('worst_10', []))
            for _ in range(remaining):
                row.extend(['', ''])
            
            row.append(result['timestamp'])
            writer.writerow(row)

def evaluate_model(model_name, model_args, label):
    """모델 평가 및 저장"""
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")
    
    
    # CUDA 메모리 최적화
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 평가 실행
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["kmmlu"],
        batch_size=8,  # OOM 방지
        device="cuda:0"
    )
    
    # 전체 평균 정확도
    overall = results['results']['kmmlu']['acc,none']
    
    # ⭐ 45개 개별 과목 점수 수집
    all_subjects = {}
    for task, metrics in results['results'].items():
        if task.startswith('kmmlu_') and 'acc,none' in metrics:
            # 대분류 제외, 개별 과목만
            if task not in ['kmmlu_stem', 'kmmlu_humss', 'kmmlu_applied_science', 'kmmlu_other', 'kmmlu']:
                subject_name = task.replace('kmmlu_', '').replace('_', ' ').title()
                all_subjects[subject_name] = metrics['acc,none']
    
    # 점수 순으로 정렬 (높은 순)
    sorted_subjects = sorted(all_subjects.items(), key=lambda x: x[1], reverse=True)
    
    # ⭐ 핵심: 최하위 10개 과목 (점수)
    worst_10_subjects = [
        {"name": name, "score": score}
        for name, score in sorted_subjects[-10:][::-1]  # 마지막 10개를 역순으로 (낮은 순)
    ]
    
    # 결과 정리
    result_data = {
        "model": label,
        "model_path": model_name,
        "overall": overall,
        "stem": results['results']['kmmlu_stem']['acc,none'],
        "humss": results['results']['kmmlu_humss']['acc,none'],
        "applied": results['results']['kmmlu_applied_science']['acc,none'],
        "other": results['results']['kmmlu_other']['acc,none'],
        "subjects": dict(sorted_subjects),  # 전체 과목
        "best": {
            "name": sorted_subjects[0][0],
            "score": sorted_subjects[0][1]
        },
        "worst_10": worst_10_subjects,  # ⭐ 최하위 10개
        "worst": {  # 하위 호환용
            "name": sorted_subjects[-1][0],
            "score": sorted_subjects[-1][1]
        },
        "timestamp": datetime.now().isoformat()
    }
    

    
    # ⭐ 콘솔에 최하위 10개 출력
    print(f"\n✅ Evaluation Complete!")
    print(f"Overall: {overall:.2%}")
    print(f"Best:    {result_data['best']['name']} ({result_data['best']['score']:.2%})")
    
    print(f"\n📉 Worst 10 Subjects (낮은 점수):")
    for i, subject in enumerate(worst_10_subjects, 1):
        print(f"  {i:2d}. {subject['name']:35s} {subject['score']:.2%}")
    
    print(f"\nTotal Subjects: {len(all_subjects)}")
    
    # 저장
    all_results = load_results()
    
    # 기존 모델 있으면 업데이트, 없으면 추가
    existing_idx = None
    for i, r in enumerate(all_results):
        if r.get('Model') == label or r.get('model_path') == model_name:
            existing_idx = i
            break
    
    if existing_idx is not None:
        all_results[existing_idx] = result_data
    else:
        all_results.append(result_data)
    
    save_results(all_results)
    print(f"Saved to {CSV_FILE}\n")
    
    return result_data

if __name__ == "__main__":
    evaluate_model(
        model_name="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
        model_args="pretrained=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B,dtype=float16",
        label="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B (16bit)"
    )