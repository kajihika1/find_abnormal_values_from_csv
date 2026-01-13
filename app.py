"""
データ異常値検出アプリケーション（Streamlit版）
ローカルフォルダのCSVファイルを読み込み、異常値を検出・表示
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from data_anomaly_detector import DataAnomalyDetector
import traceback
import json

# ページ設定
st.set_page_config(
    page_title="データ異常値検出ツール",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_csv_files(folder_path: str) -> list:
    """指定フォルダ内のCSVファイル一覧を取得"""
    try:
        path = Path(folder_path)
        if not path.exists():
            return []
        csv_files = [f.name for f in path.glob("*.csv")]
        return sorted(csv_files)
    except Exception as e:
        st.error(f"フォルダの読み込みエラー: {e}")
        return []


def convert_numpy_types(obj):
    """
    NumPy型を標準Python型に変換する
    
    Parameters:
    -----------
    obj : Any
        変換対象のオブジェクト
        
    Returns:
    --------
    Any
        標準Python型に変換されたオブジェクト
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def load_csv_file(file_path: str, header_option: str, skip_rows: int, encoding: str) -> pd.DataFrame:
    """CSVファイルを読み込み"""
    try:
        # headerオプションの設定
        if header_option == "1行目をヘッダーとして使用":
            header = 0
        elif header_option == "ヘッダーなし（自動採番）":
            header = None
        else:  # "カスタム（N行目をヘッダー）"
            header = skip_rows
        
        # skip_rowsの設定
        skiprows = skip_rows if skip_rows > 0 and header_option != "カスタム（N行目をヘッダー）" else None
        
        df = pd.read_csv(file_path, header=header, skiprows=skiprows, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        st.error(traceback.format_exc())
        return None


def display_detection_result(detector: DataAnomalyDetector, check_name: str, check_data: dict, icon: str):
    """検出結果を表示"""
    with st.expander(f"{icon} {check_name}", expanded=False):
        if isinstance(check_data, dict):
            if check_data.get('検出'):
                st.warning("⚠️ 異常を検出しました")
                
                # 詳細情報の表示
                if '詳細' in check_data:
                    detail = check_data['詳細']
                    
                    # 特別な表示処理
                    if check_name == '重複レコード':
                        dup_count = check_data.get('完全重複レコード数', 0)
                        if dup_count > 0:
                            st.metric("完全重複レコード数", f"{dup_count}件")
                        
                        col_dup = check_data.get('カラムごとの重複', {})
                        if col_dup:
                            st.write("**カラムごとの重複:**")
                            col_dup_df = pd.DataFrame(
                                list(col_dup.items()), 
                                columns=['カラム', '重複件数']
                            ).sort_values('重複件数', ascending=False)
                            st.dataframe(col_dup_df, use_container_width=True)
                    
                    elif isinstance(detail, dict) and detail:
                        # 辞書形式の詳細を表形式で表示
                        if check_name in ['統計的外れ値_Z-score', '統計的外れ値_IQR', '数値範囲']:
                            for col_name, col_info in list(detail.items())[:10]:
                                st.write(f"**{col_name}**")
                                if isinstance(col_info, dict):
                                    info_df = pd.DataFrame([col_info])
                                    st.dataframe(info_df, use_container_width=True)
                                else:
                                    st.write(col_info)
                        else:
                            st.json(detail)
                    
                    elif isinstance(detail, list):
                        # リスト形式の詳細を表示
                        for item in detail[:10]:
                            st.write(f"- {item}")
                
                # 異常データ表示ボタン
                if check_name in detector.get_available_check_types():
                    st.write("---")
                    
                    # カラム選択オプション
                    available_columns = list(detector.anomaly_indices[check_name].keys())
                    if available_columns and check_name != '重複レコード':
                        selected_column = st.selectbox(
                            "カラムを選択",
                            options=["すべて"] + available_columns,
                            key=f"col_select_{check_name}"
                        )
                        col_filter = None if selected_column == "すべて" else selected_column
                    else:
                        col_filter = None
                    
                    # サンプル数の設定
                    n_samples = st.slider(
                        "表示件数",
                        min_value=5,
                        max_value=100,
                        value=10,
                        step=5,
                        key=f"slider_{check_name}"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"📊 データを表示", key=f"show_{check_name}"):
                            anomaly_df = detector.get_anomaly_data(
                                check_name, 
                                column=col_filter, 
                                max_rows=n_samples
                            )
                            if not anomaly_df.empty:
                                st.write(f"**異常データ ({len(anomaly_df)}件)**")
                                st.dataframe(anomaly_df, use_container_width=True)
                            else:
                                st.info("異常データがありません")
                    
                    with col2:
                        if st.button(f"💾 CSVでダウンロード", key=f"download_{check_name}"):
                            anomaly_df = detector.get_anomaly_data(check_name, column=col_filter)
                            if not anomaly_df.empty:
                                csv = anomaly_df.to_csv(index=True, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 ダウンロード実行",
                                    data=csv,
                                    file_name=f"{check_name}_anomalies.csv",
                                    mime="text/csv",
                                    key=f"download_btn_{check_name}"
                                )
                            else:
                                st.info("ダウンロードするデータがありません")
            else:
                st.success("✅ 問題なし")


def main():
    """メインアプリケーション"""
    
    # ヘッダー
    st.markdown('<div class="main-header">🔍 データ異常値検出ツール</div>', unsafe_allow_html=True)
    
    # セッション状態の初期化
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # サイドバー: ファイル選択と設定
    with st.sidebar:
        st.markdown("## 📁 ファイル選択")
        
        # フォルダパス入力
        default_folder = os.getcwd()
        folder_path = st.text_input(
            "フォルダパス",
            value=default_folder,
            help="CSVファイルが格納されているフォルダのパスを入力"
        )
        
        # フォルダ内のCSVファイル一覧
        csv_files = get_csv_files(folder_path)
        
        if csv_files:
            st.success(f"✅ {len(csv_files)}個のCSVファイルを検出")
            selected_file = st.selectbox(
                "CSVファイルを選択",
                options=csv_files,
                help="読み込むCSVファイルを選択してください"
            )
        else:
            st.warning("⚠️ CSVファイルが見つかりません")
            selected_file = None
        
        st.markdown("---")
        st.markdown("## ⚙️ 読み込み設定")
        
        # ヘッダー設定
        header_option = st.selectbox(
            "ヘッダー設定",
            options=[
                "1行目をヘッダーとして使用",
                "ヘッダーなし（自動採番）",
                "カスタム（N行目をヘッダー）"
            ],
            help="CSVファイルのヘッダー行の扱いを指定"
        )
        
        # スキップ行数
        skip_rows = 0
        if header_option == "カスタム（N行目をヘッダー）":
            skip_rows = st.number_input(
                "ヘッダー行番号（0始まり）",
                min_value=0,
                max_value=100,
                value=0,
                help="ヘッダーとして使用する行番号（0始まり）"
            )
        else:
            skip_rows = st.number_input(
                "スキップする行数",
                min_value=0,
                max_value=100,
                value=0,
                help="ファイル先頭からスキップする行数"
            )
        
        # エンコーディング設定
        encoding = st.selectbox(
            "エンコーディング",
            options=["utf-8", "utf-8-sig", "shift_jis", "cp932", "euc-jp", "iso-2022-jp"],
            index=1,  # デフォルトはutf-8-sig
            help="CSVファイルの文字エンコーディング"
        )
        
        st.markdown("---")
        st.markdown("## 🎛️ 検出パラメータ")
        
        # パラメータヘルプの展開ボタン
        with st.expander("❓ パラメータの詳細説明", expanded=False):
            st.markdown("""
            ### 📊 Z-score閾値
            **意味**: データポイントが平均からどれだけ離れているかを標準偏差の倍数で表す  
            **判定**: |Z-score| > 閾値 → 外れ値
            
            | 閾値 | 厳格度 | 外れ値割合 | 用途 |
            |:----:|:------:|:----------:|:-----|
            | 2.0 | 厳しい | 約5% | データクレンジング重視 |
            | 2.5 | やや厳しい | 約1.2% | 一般的な分析 |
            | **3.0** | **標準** | **約0.3%** | **統計的に一般的（推奨）** |
            | 3.5 | 緩い | 約0.05% | 明らかな異常のみ |
            
            ---
            
            ### 📦 IQR倍率
            **意味**: 四分位範囲(Q3-Q1)を使った外れ値検出  
            **判定**: Q1-IQR×倍率 < 値 < Q3+IQR×倍率 の範囲外→外れ値
            
            | 倍率 | 厳格度 | 外れ値割合 | 用途 |
            |:----:|:------:|:----------:|:-----|
            | 1.0 | 非常に厳しい | 約15-20% | 異常が多い場合 |
            | **1.5** | **標準** | **約1-2%** | **一般的な統計分析（推奨）** |
            | 2.0 | やや緩い | 約0.5% | やや明確な異常のみ |
            | 3.0 | 緩い | 約0.1% | 極端な異常のみ |
            
            **特徴**: 正規分布でないデータにも使用可能
            
            ---
            
            ### 🔗 相関係数閾値
            **意味**: 2つの数値カラム間の線形相関の強さ（-1.0〜+1.0）  
            **判定**: |相関係数| > 閾値 → 高相関あり
            
            | 閾値 | 相関の強さ | 用途 |
            |:----:|:----------:|:-----|
            | 0.7 | やや強い | 機械学習の特徴選択 |
            | 0.8 | 強い | データ品質チェック |
            | **0.9** | **非常に強い** | **冗長性検出（推奨）** |
            | 0.95 | ほぼ完全 | 完全な重複カラム検出 |
            
            ---
            
            ### 🎯 業務別の推奨設定
            
            **データクレンジング重視**
            - Z-score: 2.5 / IQR: 1.5 / 相関: 0.85
            
            **統計分析向け（デフォルト）**
            - Z-score: 3.0 / IQR: 1.5 / 相関: 0.9
            
            **明確な異常のみ検出**
            - Z-score: 3.5 / IQR: 2.0 / 相関: 0.95
            
            **機械学習前処理**
            - Z-score: 3.0 / IQR: 1.5 / 相関: 0.7
            """)
        
        st.markdown("")  # スペース
        
        # Z-scoreの閾値
        z_threshold = st.slider(
            "Z-score閾値",
            min_value=1.5,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Z-scoreによる外れ値判定の閾値。3.0が統計的に一般的な値。"
        )
        
        # IQRの倍率
        iqr_multiplier = st.slider(
            "IQR倍率",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.5,
            help="IQR法による外れ値判定の倍率。1.5がTukeyの標準値。"
        )
        
        # 相関係数の閾値
        corr_threshold = st.slider(
            "相関係数閾値",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="高相関と判定する相関係数の閾値。0.9で非常に強い相関を検出。"
        )
        
        st.markdown("---")
        
        # データ読み込みボタン
        load_button = st.button(
            "🚀 データを読み込んで検出開始",
            type="primary",
            use_container_width=True,
            disabled=(selected_file is None)
        )
    
    # メインエリア
    if load_button and selected_file:
        file_path = os.path.join(folder_path, selected_file)
        
        with st.spinner('データを読み込んでいます...'):
            # CSVファイルの読み込み
            df = load_csv_file(file_path, header_option, skip_rows, encoding)
            
            if df is not None:
                st.session_state.df = df
                
                # データプレビュー
                st.markdown('<div class="section-header">📊 データプレビュー</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("レコード数", f"{len(df):,}")
                with col2:
                    st.metric("カラム数", f"{len(df.columns)}")
                with col3:
                    st.metric("メモリ使用量", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                st.dataframe(df.head(10), use_container_width=True)
                
                # 異常値検出の実行
                st.markdown('<div class="section-header">🔍 異常値検出中...</div>', unsafe_allow_html=True)
                
                with st.spinner('異常値を検出しています...'):
                    try:
                        detector = DataAnomalyDetector(df, name=selected_file)
                        report = detector.detect_all(
                            z_threshold=z_threshold,
                            iqr_multiplier=iqr_multiplier,
                            correlation_threshold=corr_threshold
                        )
                        
                        st.session_state.detector = detector
                        st.session_state.report = report
                        
                        st.success("✅ 異常値検出が完了しました！")
                        
                    except Exception as e:
                        st.error(f"異常値検出エラー: {e}")
                        st.error(traceback.format_exc())
    
    # 検出結果の表示
    if st.session_state.report is not None and st.session_state.detector is not None:
        detector = st.session_state.detector
        report = st.session_state.report
        
        st.markdown('<div class="section-header">📋 検出結果レポート</div>', unsafe_allow_html=True)
        
        # 基本情報
        with st.expander("📌 基本情報", expanded=True):
            basic_info = report.get('基本情報', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("レコード数", f"{basic_info.get('レコード数', 'N/A'):,}")
            with col2:
                st.metric("カラム数", f"{basic_info.get('カラム数', 'N/A')}")
            with col3:
                st.metric("メモリ使用量", f"{basic_info.get('メモリ使用量_MB', 0):.2f} MB")
            with col4:
                anomaly_count = len([k for k, v in report.items() 
                                   if isinstance(v, dict) and v.get('検出')])
                st.metric("異常検出項目数", f"{anomaly_count}")
        
        # 各検出項目の結果
        checks = [
            ('NULL値', '❌'),
            ('重複レコード', '📋'),
            ('データ型', '🔤'),
            ('統計的外れ値_Z-score', '📊'),
            ('統計的外れ値_IQR', '📊'),
            ('カテゴリカルデータ', '🏷️'),
            ('文字列異常', '📝'),
            ('空白スペース', '⬜'),
            ('数値範囲', '🔢'),
            ('時系列異常', '📅'),
            ('相関異常', '🔗')
        ]
        
        st.markdown('<div class="section-header">🔎 詳細検出結果</div>', unsafe_allow_html=True)
        
        for check_name, icon in checks:
            if check_name in report:
                check_data = report[check_name]
                display_detection_result(detector, check_name, check_data, icon)
        
        # 全体レポートのダウンロード
        st.markdown("---")
        st.markdown("### 📥 レポートのダウンロード")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # NumPy型を標準Python型に変換してからJSON化
            report_converted = convert_numpy_types(report)
            report_json = json.dumps(report_converted, ensure_ascii=False, indent=2)
            st.download_button(
                label="📄 レポート（JSON形式）",
                data=report_json,
                file_name="anomaly_report.json",
                mime="application/json"
            )
        
        with col2:
            if st.session_state.df is not None:
                csv = st.session_state.df.to_csv(index=True, encoding='utf-8-sig')
                st.download_button(
                    label="📊 元データ（CSV形式）",
                    data=csv,
                    file_name="original_data.csv",
                    mime="text/csv"
                )
    
    elif st.session_state.detector is None:
        # 初期表示メッセージ
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 使い方
        
        1. **サイドバー**でフォルダパスを指定
        2. **CSVファイル**を選択
        3. **読み込み設定**（ヘッダー、エンコーディング等）を調整
        4. **検出パラメータ**を必要に応じて変更
        5. **「データを読み込んで検出開始」**ボタンをクリック
        
        ### 📊 検出項目
        
        - NULL値の検出
        - 重複レコードの検出
        - データ型の不整合
        - 統計的外れ値（Z-score、IQR法）
        - カテゴリカルデータの異常
        - 文字列の異常パターン
        - 空白・スペースの問題
        - 数値範囲の異常
        - 時系列データの異常
        - 相関関係の異常
        """)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
