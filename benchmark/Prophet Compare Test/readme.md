## Shape Test for MSFR - Prototype
- Dataset : Same as testing MSFR(electricity usage per 15minutes)
- Reason why we should do this Benchmark : MSFR와 비슷한 선행연구 프로젝트, Prophet이랑 성능 비교가 필요하다고 판단(MAE, L-BFGS 사용 예정, Huber는 강화 학습 쪽에 더 어울린다고 해서)
- Tester : 함태준(rrayy-25809)
- [make_plot_by_prophet.py](https://github.com/tatatommy6/multi-seasonal-fourier-regression/blob/main/benchmark/Prophet%20Compare%20Test/make_plot_by_porphet.py)는 prophet을 이용하여 사용자가 원하는 가구를 입력받은 후 그 가구를 예측한 후 그래프로 표현하는 코드지 MSFR과 직접적으로 비교하지 않음.(즉, 그래프 차이는 tester가 직접 눈으로 비교해야함.)
