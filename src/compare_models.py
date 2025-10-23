# compare_models.py
from tabulate import tabulate
import json  
import os

JSON_LEADERBOARD = 'kmmlu_leaderboard.json'  

def compare_models():
    """저장된 모든 모델 비교"""
    
    if not os.path.exists(JSON_LEADERBOARD):
        print("❌ No results found! Run evaluate_model.py first.")
        return
    
    # JSON 읽기 
    with open(JSON_LEADERBOARD, 'r', encoding='utf-8') as f:
        results = json.load(f)  
    
    if not results:
        print("❌ No valid results found!")
        return
    
    # overall 점수로 정렬
    results.sort(key=lambda x: x['overall'], reverse=True)
    
    print("\n" + "="*140)
    print(f"{'KMMLU MODEL COMPARISON':^140}")
    print("="*140)
    print(f"Total Models: {len(results)}\n")
    
    table_data = []
    for i, r in enumerate(results, 1):
        rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        
        table_data.append([
            rank,
            r['model'][:30],
            f"{r['overall']:.1%}",
            f"{r['best']['name'][:15]}\n({r['best']['score']:.1%})",
            f"{r['worst']['name'][:15]}\n({r['worst']['score']:.1%})",
            f"{r['stem']:.1%}",
            f"{r['humss']:.1%}",
            f"{r['applied']:.1%}",
            f"{r['other']:.1%}",
            r.get('elapsed_time', 'N/A'),
            r.get('batch_size', 'N/A'),
            r.get('precision', 'N/A')
        ])
    
    print(tabulate(
        table_data,
        headers=["Rank", "Model", "Overall", "Best Subject", "Worst Subject", 
                 "STEM", "HUMSS", "Applied", "Other", "Time", "Batch", "Precision"],
        tablefmt="grid"
    ))
    
    if len(results) > 1:
        best = results[0]
        worst = results[-1]
        gap = (best['overall'] - worst['overall']) * 100
        
        print(f"\n📊 Stats:")
        print(f"  Best:  {best['model']} ({best['overall']:.2%})")
        print(f"  Worst: {worst['model']} ({worst['overall']:.2%})")
        print(f"  Gap:   {gap:.1f}pp\n")
    
    print(f"\n💡 Tip: Check '{JSON_LEADERBOARD}' for detailed model data\n")

if __name__ == "__main__":
    compare_models()