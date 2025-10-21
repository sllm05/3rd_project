# compare_models.py
# 평가된 모든 모델들의 결과를 비교하여 순위표로 출력하는 스크립트
# evaluate_model.py로 평가한 후 이 스크립트를 실행하면 예쁜 표를 볼 수 있음

# === 필요한 라이브러리 임포트 ===
from tabulate import tabulate  # 데이터를 예쁜 ASCII 테이블로 만들어주는 라이브러리
                                # pip install tabulate 로 설치 필요
import json  # JSON 파일 읽기용 표준 라이브러리
import os  # 파일 시스템 작업용 (파일 존재 확인)

# === 전역 상수 정의 ===
RESULTS_FILE = 'kmmlu_results.json'  # 모델 평가 결과가 저장된 파일 경로
                                     # evaluate_model.py가 생성한 파일

def compare_models():
    """저장된 모든 모델 비교"""
    
    # === 1단계: 파일 존재 여부 확인 ===
    # os.path.exists(): 파일이 실제로 있는지 확인하는 함수
    if not os.path.exists(RESULTS_FILE):
        # 파일이 없으면 (아직 모델을 평가하지 않았으면)
        print("❌ No results found! Run evaluate_model.py first.")
        # 사용자에게 먼저 evaluate_model.py를 실행하라고 안내
        return  # 함수 종료 (더 이상 진행 불가)
    
    # === 2단계: JSON 파일에서 결과 읽기 ===
    with open(RESULTS_FILE, 'r') as f:
        # 'r' 모드: 읽기 전용으로 파일 열기
        # with 문: 파일을 자동으로 닫아줌 (메모리 누수 방지)
        results = json.load(f)
        # json.load(): JSON 텍스트를 파이썬 리스트/딕셔너리로 변환
        # results는 여러 모델의 평가 결과를 담은 리스트
        # 예: [{"model": "Llama-3", "overall": 0.75, ...}, {"model": "GPT-4", ...}]
    
    # === 3단계: 빈 결과 확인 ===
    if not results:
        # 파일은 있지만 내용이 비어있는 경우
        print("❌ No results found!")
        return
    
    # === 4단계: 점수순으로 정렬 ===
    # sort(): 리스트를 제자리에서 정렬 (원본 수정)
    # key=lambda x: x['overall']: 각 딕셔너리의 'overall' 값을 기준으로 정렬
    #   - lambda는 익명 함수 (간단한 일회용 함수)
    #   - x는 리스트의 각 요소(딕셔너리)
    # reverse=True: 내림차순 정렬 (높은 점수부터)
    results.sort(key=lambda x: x['overall'], reverse=True)
    # 정렬 후: 1등 모델이 results[0], 꼴등이 results[-1]
    
    # === 5단계: 테이블 헤더 출력 ===
    print("\n" + "="*120)  # 120개의 '=' 문자로 구분선 그리기
    
    # f-string의 중앙 정렬 포맷팅
    # {:^120}는 전체 너비 120자 중앙에 텍스트 배치
    print(f"{'KMMLU MODEL COMPARISON':^120}")
    
    print("="*120)  # 아래 구분선
    
    # 총 몇 개의 모델이 평가되었는지 표시
    print(f"Total Models: {len(results)}\n")  # len(): 리스트 길이
    
    # === 6단계: 테이블 데이터 준비 ===
    table_data = []  # 테이블의 각 행을 담을 빈 리스트
    
    # enumerate(results, 1): 리스트를 순회하며 인덱스도 함께 가져옴
    #   - 1부터 시작 (기본값은 0)
    #   - i: 순위 (1, 2, 3, ...)
    #   - r: 각 모델의 결과 딕셔너리
    for i, r in enumerate(results, 1):
        
        # === 순위 표시 만들기 ===
        # 삼항 연산자를 중첩하여 1~3등은 메달, 나머지는 숫자로 표시
        rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        # 논리:
        #   - i == 1이면 "🥇"
        #   - 아니고 i == 2이면 "🥈"
        #   - 아니고 i == 3이면 "🥉"
        #   - 그것도 아니면 "#4", "#5" 형태
        
        # === 테이블 한 행 데이터 구성 ===
        table_data.append([
            # 각 리스트 요소가 테이블의 한 열(column)이 됨
            
            rank,  # 1열: 순위 (메달 또는 번호)
            
            r['model'][:35],  # 2열: 모델명 (35자로 제한)
            # 슬라이싱 [:35]으로 너무 긴 이름 잘라내기
            # 예: "Very-Long-Model-Name-That-Goes-On..." → "Very-Long-Model-Name-That-Go..."
            
            f"{r['overall']:.1%}",  # 3열: 전체 점수
            # :.1%는 소수점 1자리 백분율 포맷
            # 예: 0.7523 → "75.2%"
            
            # 4열: 최고 성적 과목 (2줄로 표시)
            f"{r['best']['name'][:18]}\n({r['best']['score']:.1%})",
            # [:18]로 과목명 길이 제한
            # \n으로 줄바꿈 (과목명과 점수를 위아래로)
            # 예: "Computer Science
            #      (82.5%)"
            
            # 5열: 최저 성적 과목 (2줄로 표시)
            f"{r['worst']['name'][:18]}\n({r['worst']['score']:.1%})",
            # 구조는 위와 동일
            
            # 6~9열: 4개 대분류 카테고리 점수
            f"{r['stem']:.1%}",      # STEM (과학/기술/공학/수학)
            f"{r['humss']:.1%}",     # HUMSS (인문/사회과학)
            f"{r['applied']:.1%}",   # Applied Science (응용과학)
            f"{r['other']:.1%}"      # Other (기타)
        ])
        # append(): 리스트 끝에 새 항목 추가
    
    # === 7단계: 테이블 출력 ===
    # tabulate(): 2차원 리스트를 보기 좋은 테이블로 변환
    print(tabulate(
        table_data,  # 실제 데이터 (각 행의 리스트)
        
        # headers: 각 열의 제목
        headers=["Rank", "Model", "Overall", "Best Category", "Worst Category", 
                 "STEM", "HUMSS", "Applied", "Other"],
        
        # tablefmt: 테이블 스타일
        #   - "grid": 모든 셀을 박스로 둘러쌈 (├─┼─┤ 같은 문자 사용)
        #   - 다른 옵션: "simple", "pipe", "html" 등
        tablefmt="grid"
    ))
    
    # === 8단계: 통계 정보 계산 및 출력 ===
    # 모델이 2개 이상일 때만 통계 의미가 있음
    if len(results) > 1:
        
        # 1등과 꼴등 모델 가져오기
        best = results[0]   # 정렬했으므로 첫 번째가 1등
        worst = results[-1]  # 마지막이 꼴등 ([-1]은 리스트의 마지막 요소)
        
        # 점수 차이 계산 (퍼센트 포인트)
        # 예: 0.85 - 0.60 = 0.25 → 25%p
        gap = (best['overall'] - worst['overall']) * 100
        # * 100: 0.25 → 25.0
        
        # 통계 섹션 출력
        print(f"\n📊 Stats:")  # 제목
        
        # 최고 모델 정보
        print(f"  Best:  {best['model']} ({best['overall']:.2%})")
        # :.2%는 소수점 2자리 백분율
        # 예: "Best:  Llama-3-Ko-8B (75.23%)"
        
        # 최저 모델 정보
        print(f"  Worst: {worst['model']} ({worst['overall']:.2%})")
        # 예: "Worst: Small-Model (45.67%)"
        
        # 점수 차이
        print(f"  Gap:   {gap:.1f}pp\n")
        # pp: percentage point (퍼센트 포인트)
        # :.1f는 소수점 1자리 일반 숫자 포맷
        # 예: "Gap:   29.6pp"

# === 메인 실행 블록 ===
# 이 파일을 직접 실행했을 때만 아래 코드 실행
# 다른 파일에서 import 할 때는 실행 안 됨
if __name__ == "__main__":
    compare_models()  # 비교 함수 호출
    
# === 사용 방법 ===
# 1. 먼저 evaluate_model.py로 모델들을 평가
# 2. 그 다음 이 파일 실행: python compare_models.py
# 3. 순위표가 출력됨!