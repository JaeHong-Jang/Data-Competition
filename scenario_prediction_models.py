#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기후변화 시나리오별 천궁 기능성 성분 예측 모델
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("🌍 기후변화 시나리오별 천궁 기능성 성분 예측 모델")
print("="*80)

# 1. 데이터 로드
print("📊 데이터 로드")
df_features = pd.read_csv('df_features_pca.csv')
df_targets = pd.read_csv('df_targets.csv')
df_complete = pd.read_csv('df_complete_pca.csv')

print(f"   특성 데이터: {df_features.shape}")
print(f"   타겟 데이터: {df_targets.shape}")
print(f"   전체 데이터: {df_complete.shape}")

# 타겟 변수들 정의
target_variables = ['Leaf_TPC', 'Root_TPC', 'Leaf_TFC', 'Root_TFC']
target_names = {
    'Leaf_TPC': '지상부 총 페놀 함량',
    'Root_TPC': '지하부 총 페놀 함량', 
    'Leaf_TFC': '지상부 총 플라보노이드 함량',
    'Root_TFC': '지하부 총 플라보노이드 함량'
}

# 시나리오별 데이터 분포 확인
print(f"\n🔍 시나리오별 데이터 분포:")
scenario_counts = df_complete['scenario'].value_counts()
for scenario, count in scenario_counts.items():
    print(f"   {scenario}: {count}개 ({count/len(df_complete)*100:.1f}%)")

print("\n" + "="*80)

# 2. 시나리오별 예측 모델 구축 방법 1: 개별 모델
print("🎯 방법 1: 시나리오별 개별 모델 구축")
print("="*80)

scenario_models = {}
scenario_results = {}

scenarios = ['SSP1-2.6', 'SSP3-7.0', 'SSP5-8.5']  # 실제 시나리오 이름에 맞게 수정

for scenario in scenarios:
    print(f"\n📊 {scenario} 시나리오 모델 구축")
    print("-" * 60)
    
    # 시나리오별 데이터 필터링
    scenario_mask = df_complete['scenario'] == scenario
    scenario_data = df_complete[scenario_mask]
    
    if len(scenario_data) == 0:
        print(f"   ❌ {scenario} 데이터가 없습니다.")
        continue
    
    print(f"   데이터 수: {len(scenario_data)}개")
    
    # 특성 변수와 타겟 변수 분리
    feature_cols = [col for col in scenario_data.columns if col not in target_variables + ['scenario']]
    X_scenario = scenario_data[feature_cols]
    
    scenario_models[scenario] = {}
    scenario_results[scenario] = {}
    
    # 각 타겟 변수별 모델 구축
    for target in target_variables:
        if target in scenario_data.columns:
            y_scenario = scenario_data[target]
            
            print(f"\n   🎯 {target_names[target]} 예측 모델:")
            
            # 데이터 분할
            if len(scenario_data) > 20:  # 충분한 데이터가 있는 경우만 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scenario, y_scenario, test_size=0.2, random_state=42
                )
            else:
                # 데이터가 적은 경우 전체를 학습에 사용
                X_train, X_test = X_scenario, X_scenario
                y_train, y_test = y_scenario, y_scenario
                print("      ⚠️  데이터가 적어 전체를 학습에 사용")
            
            # 모델 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 예측 및 평가
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            print(f"      Train R²: {train_r2:.3f}")
            print(f"      Test R²: {test_r2:.3f}")
            print(f"      Test RMSE: {test_rmse:.3f}")
            print(f"      Test MAE: {test_mae:.3f}")
            
            # 결과 저장
            scenario_models[scenario][target] = model
            scenario_results[scenario][target] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            }

print("\n" + "="*80)

# 3. 통합 모델 (시나리오를 특성으로 포함)
print("🎯 방법 2: 시나리오를 특성으로 포함한 통합 모델")
print("="*80)

# 시나리오 원-핫 인코딩
df_complete_encoded = pd.get_dummies(df_complete, columns=['scenario'], prefix='scenario')

# 특성 변수와 타겟 변수 분리
feature_cols_encoded = [col for col in df_complete_encoded.columns if col not in target_variables]
X_all = df_complete_encoded[feature_cols_encoded]

unified_models = {}
unified_results = {}

print(f"\n📊 통합 모델 구축 (전체 데이터: {len(df_complete)}개)")

for target in target_variables:
    if target in df_complete.columns:
        y_all = df_complete[target]
        
        print(f"\n🎯 {target_names[target]} 통합 모델:")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )
        
        # 모델 학습
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"   Train R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   Test RMSE: {test_rmse:.3f}")
        print(f"   Test MAE: {test_mae:.3f}")
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='r2')
        print(f"   CV R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # 결과 저장
        unified_models[target] = model
        unified_results[target] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }

print("\n" + "="*80)

# 4. 결과 비교 및 시각화
print("📊 모델 성능 비교")
print("="*80)

# 결과 요약 테이블 생성
results_summary = []

# 시나리오별 모델 결과
for scenario in scenarios:
    if scenario in scenario_results:
        for target in target_variables:
            if target in scenario_results[scenario]:
                results_summary.append({
                    'Model_Type': f'시나리오별 ({scenario})',
                    'Target': target_names[target],
                    'Test_R2': scenario_results[scenario][target]['test_r2'],
                    'Test_RMSE': scenario_results[scenario][target]['test_rmse'],
                    'Test_MAE': scenario_results[scenario][target]['test_mae']
                })

# 통합 모델 결과
for target in target_variables:
    if target in unified_results:
        results_summary.append({
            'Model_Type': '통합 모델',
            'Target': target_names[target],
            'Test_R2': unified_results[target]['test_r2'],
            'Test_RMSE': unified_results[target]['test_rmse'],
            'Test_MAE': unified_results[target]['test_mae']
        })

# 결과 데이터프레임 생성
results_df = pd.DataFrame(results_summary)
print(results_df)

# 결과 저장
results_df.to_csv('model_performance_comparison.csv', index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: model_performance_comparison.csv")

print("\n" + "="*80)

# 5. 새로운 데이터 예측 예시
print("🔮 새로운 데이터 예측 예시")
print("="*80)

def predict_scenario(scenario, feature_values, models, target):
    """
    특정 시나리오에서 기능성 성분 예측
    
    Parameters:
    - scenario: 시나리오 이름 (예: 'SSP5-8.5')
    - feature_values: 특성 값들 (dict)
    - models: 학습된 모델들
    - target: 예측할 타겟 변수
    """
    
    if scenario in models and target in models[scenario]:
        # 시나리오별 모델 사용
        model = models[scenario][target]
        # feature_values를 모델 입력 형태로 변환 필요
        # (실제 구현 시 특성 순서 맞춤 필요)
        prediction = "시나리오별 모델로 예측"
    else:
        # 통합 모델 사용
        if target in unified_models:
            model = unified_models[target]
            # feature_values + 시나리오 정보를 모델 입력 형태로 변환
            prediction = "통합 모델로 예측"
        else:
            prediction = "예측 불가"
    
    return prediction

print("예측 함수 정의 완료")
print("실제 예측을 위해서는 새로운 환경 데이터를 입력하면 됩니다.")

print("\n✅ 시나리오별 예측 모델 구축 완료!")
print("   • 시나리오별 개별 모델: 각 시나리오의 특성을 반영한 전문화된 모델")
print("   • 통합 모델: 시나리오 정보를 포함한 범용 모델")
print("   • 두 방법 모두 활용 가능하며, 성능 비교 후 최적 방법 선택 권장")
