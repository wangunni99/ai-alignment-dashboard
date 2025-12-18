import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import altair as alt
import io
import re

# SBERT 모델 로딩 (캐싱을 사용하여 성능 최적화)
# Streamlit 환경에서 모델을 한 번만 로드하도록 st.cache_resource 사용
@st.cache_resource
def load_sbert_model():
    try:
        # 사용자가 요청한 한국어 특화 SBERT 모델
        model_name = 'jhgan/ko-sbert-multitask'
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        st.error("🚨 **오류:** 'sentence-transformers' 라이브러리가 설치되지 않았습니다. `requirements.txt`를 확인하고 설치해주세요.")
        return None
    except Exception as e:
        st.error(f"🚨 **오류:** SBERT 모델 로딩 중 문제가 발생했습니다: {e}")
        return None

# 1. 데이터 로딩 및 전처리
def load_and_preprocess_data(uploaded_file):
    """엑셀 파일을 로드하고 데이터를 전처리합니다."""
    if uploaded_file is None:
        return None, None

    try:
        # openpyxl 엔진을 사용하여 엑셀 파일 로드
        xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
        
        # 시트 로드
        df_business = pd.read_excel(xls, '사업')
        df_tech = pd.read_excel(xls, '기술')

        # 데이터 정규화 및 결합
        df_business = preprocess_data(df_business, '사업')
        df_tech = preprocess_data(df_tech, '기술')

        return df_business, df_tech

    except ValueError as e:
        st.error(f"🚨 **오류:** 엑셀 파일에 '사업' 또는 '기술' 시트가 없습니다. 시트 이름을 확인해주세요. ({e})")
        return None, None
    except Exception as e:
        st.error(f"🚨 **오류:** 파일 처리 중 예상치 못한 오류가 발생했습니다: {e}")
        return None, None

def preprocess_data(df, project_type):
    """개별 데이터프레임 전처리 (조직명 정규화, 임베딩 텍스트 생성)"""
    
    # 컬럼명 통일
    df.columns = [
        '프로젝트명', '설명', 'PO 조직', '유관 조직'
    ]
    
    # 데이터프레임에 프로젝트명과 설명이 없는 경우 처리
    if '프로젝트명' not in df.columns or '설명' not in df.columns:
        st.error(f"🚨 **오류:** '{project_type}' 시트에 '프로젝트명' 또는 '설명' 컬럼이 없습니다.")
        return pd.DataFrame()

    # 조직명 결측값 처리: 'nan' 노드 방지를 위해 '미지정'으로 대체
    df['PO 조직'] = df['PO 조직'].fillna('미지정')
    df['유관 조직'] = df['유관 조직'].fillna('') # 유관 조직은 리스트로 변환되므로 빈 문자열로 처리

    # 조직명 정규화: 쉼표(,) 또는 줄바꿈(\n)으로 분리 후 공백 제거
    def normalize_orgs(org_str):
        if pd.isna(org_str) or org_str == '':
            return []
        # 쉼표, 줄바꿈, 세미콜론 등을 구분자로 사용
        org_list = re.split(r'[,\n;]', str(org_str))
        # 각 조직명에서 앞뒤 공백 제거
        return [org.strip() for org in org_list if org.strip()]

    df['유관 조직_list'] = df['유관 조직'].apply(normalize_orgs)
    
    # 임베딩에 사용할 텍스트 생성: '프로젝트명 + 설명'
    df['embedding_text'] = df['프로젝트명'].fillna('') + " [설명]: " + df['설명'].fillna('')
    
    # 프로젝트 고유 ID 생성
    df['project_id'] = [f"{project_type}_{i}" for i in range(len(df))]
    df['project_type'] = project_type
    
    return df

# 2. 핵심 분석 로직 (Backend)
def get_embeddings(texts, model):
    """SBERT 모델을 사용하여 텍스트를 임베딩합니다."""
    if model is None:
        return np.array([])
    
    with st.spinner("⏳ 프로젝트 텍스트를 벡터화하는 중... (SBERT 모델 사용)"):
        # 텍스트가 비어있는 경우를 대비하여 필터링
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            return np.array([])
            
        embeddings = model.encode(valid_texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def calculate_similarity(business_embeddings, tech_embeddings):
    """사업 프로젝트와 기술 프로젝트 간의 코사인 유사도를 계산합니다."""
    if business_embeddings.size == 0 or tech_embeddings.size == 0:
        return np.array([[]])
        
    with st.spinner("📐 프로젝트 간 유사도를 계산하는 중..."):
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(business_embeddings, tech_embeddings)
    return similarity_matrix

def get_matches(df_business, df_tech, similarity_matrix, threshold):
    """유사도 임계값을 기준으로 매칭된 프로젝트 목록을 생성합니다."""
    
    matches = []
    
    # 유사도 행렬을 순회하며 임계값 이상의 매칭을 찾음
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                matches.append({
                    '사업_ID': df_business.iloc[i]['project_id'],
                    '사업_프로젝트명': df_business.iloc[i]['프로젝트명'],
                    '사업_PO_조직': df_business.iloc[i]['PO 조직'],
                    '기술_ID': df_tech.iloc[j]['project_id'],
                    '기술_프로젝트명': df_tech.iloc[j]['프로젝트명'],
                    '기술_PO_조직': df_tech.iloc[j]['PO 조직'],
                    '유사도': similarity
                })
    
    df_matches = pd.DataFrame(matches)
    return df_matches

# 3. 대시보드 UI/UX 구현
def display_kpis(df_business, df_tech, df_matches):
    """메인 상단 KPI 요약 정보를 표시합니다."""
    
    total_business = len(df_business)
    total_tech = len(df_tech)
    matched_count = len(df_matches)
    
    # 매칭된 사업 프로젝트 수 (중복 제거)
    matched_business_count = df_matches['사업_ID'].nunique() if not df_matches.empty else 0
    
    # Alignment Rate (%) 계산
    if total_business > 0:
        alignment_rate = (matched_business_count / total_business) * 100
    else:
        alignment_rate = 0
        
    st.subheader("📊 AI 프로젝트 얼라인먼트 현황 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("총 사업 프로젝트 수", f"{total_business}건")
    col2.metric("총 기술 프로젝트 수", f"{total_tech}건")
    col3.metric("매칭된 연결 건수", f"{matched_count}건")
    col4.metric("Alignment Rate (%)", f"{alignment_rate:.1f}%", 
                help="기술 프로젝트와 매칭된 사업 프로젝트의 비율")

def create_network_map(df_matches):
    """PyVis와 NetworkX를 사용하여 조직 간 협업 네트워크 맵을 생성합니다."""
    
    # NetworkX 그래프 생성
    G = nx.Graph()
    
    # 노드 추가: PO 조직
    all_orgs = pd.concat([df_matches['사업_PO_조직'], df_matches['기술_PO_조직']]).unique()
    for org in all_orgs:
        G.add_node(org, group='조직')

    # 엣지 추가: 매칭된 프로젝트를 기반으로 조직 간 협업 관계를 엣지로 표현
    for index, row in df_matches.iterrows():
        org1 = row['사업_PO_조직']
        org2 = row['기술_PO_조직']
        similarity = row['유사도']
        
        # 조직이 다를 경우에만 엣지 추가 (자기 자신과의 연결 제외)
        if org1 != org2:
            # 엣지 가중치 (유사도)를 사용하여 협업 강도 표현
            if G.has_edge(org1, org2):
                # 이미 엣지가 있다면 가중치 업데이트 (합산)
                G[org1][org2]['title'] += f" - {row['사업_프로젝트명']} - {row['기술_프로젝트명']} ({similarity:.2f})"
                G[org1][org2]['weight'] += similarity
                G[org1][org2]['label'] = f"{G[org1][org2]['weight']:.1f}"
            else:
                G.add_edge(org1, org2, 
                           weight=similarity, 
                           title=f"- {row['사업_프로젝트명']} - {row['기술_프로젝트명']} ({similarity:.2f})",
                           label=f"{similarity:.1f}")

    # --- 시각화 개선 로직 ---
    
    # 1. 노드 크기: 연결 중심성(Degree Centrality) 반영
    if G.number_of_nodes() > 0:
        # 연결 중심성 계산
        degree_centrality = nx.degree_centrality(G)
        
        # 노드 크기 업데이트: 중심성에 비례하여 크기 설정 (최소 10, 최대 50)
        max_centrality = max(degree_centrality.values()) if degree_centrality else 1
        for node in G.nodes():
            centrality = degree_centrality.get(node, 0)
            # 크기 변화 폭을 크게 설정
            size = 10 + (centrality / max_centrality) * 40 
            G.nodes[node]['size'] = size
            G.nodes[node]['title'] = f"조직: {node}  
연결 중심성: {centrality:.2f}  
총 협업 강도: {G.degree(node, weight='weight'):.1f}"
    
    # 2. PyVis 네트워크 생성
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local')
    
    # 3. 물리 엔진 설정 강화 (군집화 개선)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,  // 인력 강화 (노드들이 더 잘 뭉침)
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        }
      }
    }
    """)
    
    # NetworkX 그래프를 PyVis로 변환 (수동 변환으로 PyVis 호환성 문제 해결)
    for node in G.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        net.add_node(str(node_id), 
                     label=str(node_id), 
                     title=node_data.get('title', str(node_id)), 
                     group=node_data.get('group'), 
                     size=node_data.get('size', 10))
    
    for edge in G.edges(data=True):
        net.add_edge(str(edge[0]), str(edge[1]), 
                     value=edge[2].get('weight'), 
                     title=edge[2].get('title'), 
                     label=edge[2].get('label'))
    
    # HTML 파일로 저장
    net.save_graph("network_map.html")
    
    # Streamlit에 HTML 렌더링
    try:
        import streamlit.components.v1 as components
        with open("network_map.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=650)
    except Exception as e:
        st.error(f"PyVis 렌더링 오류: {e}")
        st.info("PyVis 네트워크 맵을 렌더링하려면 `streamlit.components.v1`이 필요합니다.")

def display_gap_analysis(df_business, df_tech, df_matches):
    """갭 분석 (Tech Gap, Tech Push) 결과를 표시합니다."""
    
    # 1. Tech Gap: 기술 과제와 매칭되지 않은 사업 프로젝트
    matched_business_ids = set(df_matches['사업_ID']) if not df_matches.empty else set()
    df_tech_gap = df_business[~df_business['project_id'].isin(matched_business_ids)]
    
    st.markdown("#### 🔴 Tech Gap (기술 지원 필요)")
    st.info(f"총 {len(df_tech_gap)}건의 사업 프로젝트가 매칭되는 기술 프로젝트를 찾지 못했습니다.")
    if not df_tech_gap.empty:
        st.dataframe(df_tech_gap[['프로젝트명', 'PO 조직', '설명']], use_container_width=True)

    st.markdown("---")

    # 2. Tech Push: 사업 과제와 매칭되지 않은 기술 프로젝트
    matched_tech_ids = set(df_matches['기술_ID']) if not df_matches.empty else set()
    df_tech_push = df_tech[~df_tech['project_id'].isin(matched_tech_ids)]
    
    st.markdown("#### 🟢 Tech Push (사업화 필요)")
    st.info(f"총 {len(df_tech_push)}건의 기술 프로젝트가 매칭되는 사업 프로젝트를 찾지 못했습니다.")
    if not df_tech_push.empty:
        st.dataframe(df_tech_push[['프로젝트명', 'PO 조직', '설명']], use_container_width=True)

def display_workload(df_business, df_tech):
    """조직별 프로젝트 현황 (Workload)을 막대 그래프로 표시합니다."""
    
    # PO 조직별 프로젝트 수 집계
    df_business_po = df_business.groupby('PO 조직').size().reset_index(name='사업_프로젝트_수')
    df_tech_po = df_tech.groupby('PO 조직').size().reset_index(name='기술_프로젝트_수')
    
    # 데이터 병합
    df_workload = pd.merge(df_business_po, df_tech_po, on='PO 조직', how='outer').fillna(0)
    
    # Wide format을 Long format으로 변환 (Altair 시각화를 위해)
    df_workload_long = pd.melt(df_workload, id_vars=['PO 조직'], 
                               value_vars=['사업_프로젝트_수', '기술_프로젝트_수'],
                               var_name='프로젝트_유형', value_name='프로젝트_수')
    
    st.markdown("#### 📈 조직별 프로젝트 현황 (Workload)")
    
    # Altair 막대 그래프 생성
    chart = alt.Chart(df_workload_long).mark_bar().encode(
        # x축: 프로젝트 수 (합계)
        x=alt.X('프로젝트_수:Q', title='프로젝트 수 (합계)'),
        # y축: PO 조직
        y=alt.Y('PO 조직:N', sort='-x', title='PO 조직'),
        # 색상: 프로젝트 유형별 구분
        color=alt.Color('프로젝트_유형:N', title='유형'),
        # 툴팁 설정
        tooltip=['PO 조직', '프로젝트_유형', '프로젝트_수']
    ).properties(
        title="조직별 사업 및 기술 프로젝트 담당 현황"
    ).interactive() # 줌/팬 기능 활성화
    
    st.altair_chart(chart, use_container_width=True)

def to_csv(df):
    """데이터프레임을 CSV 형식으로 변환합니다."""
    # 인코딩 문제 방지를 위해 BOM이 포함된 UTF-8로 인코딩
    return df.to_csv(index=False, encoding='utf-8-sig')

def main():
    """Streamlit 애플리케이션의 메인 함수입니다."""
    st.set_page_config(
        page_title="AI 프로젝트 얼라인먼트 대시보드",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 AI 프로젝트 얼라인먼트 대시보드")
    st.markdown("사업 프로젝트와 기술 프로젝트 간의 유사도를 분석하여 조직 간 협업 시너지를 시각화합니다.")

    # --- 사이드바 (Settings) ---
    st.sidebar.header("⚙️ 설정")
    
    # 엑셀 파일 업로드 위젯
    uploaded_file = st.sidebar.file_uploader(
        "엑셀 파일(.xlsx) 업로드", 
        type=['xlsx'],
        help="시트 1: '사업', 시트 2: '기술'이 포함된 엑셀 파일을 업로드하세요."
    )
    
    # 유사도 임계값 조절 슬라이더
    threshold = st.sidebar.slider(
        "유사도 임계값 (Threshold)", 
        min_value=0.40, 
        max_value=0.95, 
        value=0.60, 
        step=0.01,
        help="이 값 이상인 프로젝트만 '연결된 프로젝트'로 간주합니다."
    )
    
    # --- 메인 로직 실행 ---
    
    # 엑셀 파일 로드 및 전처리
    df_business, df_tech = load_and_preprocess_data(uploaded_file)

    if df_business is None or df_tech is None or df_business.empty or df_tech.empty:
        st.info("⬆️ 왼쪽 사이드바에서 엑셀 파일을 업로드하여 분석을 시작하세요.")
        return

    # SBERT 모델 로드
    model = load_sbert_model()
    if model is None:
        return

    # 텍스트 임베딩
    business_embeddings = get_embeddings(df_business['embedding_text'].tolist(), model)
    tech_embeddings = get_embeddings(df_tech['embedding_text'].tolist(), model)
    
    if business_embeddings.size == 0 or tech_embeddings.size == 0:
        st.warning("임베딩 데이터가 비어있습니다. 프로젝트 데이터(프로젝트명, 설명)를 확인해주세요.")
        return

    # 유사도 계산
    similarity_matrix = calculate_similarity(business_embeddings, tech_embeddings)
    
    # 매칭 결과 추출
    df_matches = get_matches(df_business, df_tech, similarity_matrix, threshold)

    # --- 메인 콘텐츠 ---
    
    # 1. KPI 요약
    display_kpis(df_business, df_tech, df_matches)
    st.markdown("---")

    # 2. 탭 구성
    tab1, tab2, tab3 = st.tabs(["🌐 네트워크 맵", "🔍 갭 분석", "💼 리소스 현황"])

    with tab1:
        st.header("조직 간 협업 네트워크 맵")
        if df_matches.empty:
            st.warning("매칭된 프로젝트가 없습니다. 임계값을 낮추거나 데이터를 확인해주세요.")
        else:
            create_network_map(df_matches)

    with tab2:
        st.header("프로젝트 얼라인먼트 갭 분석")
        display_gap_analysis(df_business, df_tech, df_matches)

    with tab3:
        st.header("조직별 프로젝트 리소스 현황")
        display_workload(df_business, df_tech)

    # 3. 매칭 결과 상세 테이블 및 다운로드
    st.markdown("---")
    st.subheader("📋 매칭 결과 상세")
    
    if df_matches.empty:
        st.info("현재 임계값(%.2f)에서는 매칭된 프로젝트가 없습니다." % threshold)
    else:
        # 유사도 순으로 정렬
        df_matches_sorted = df_matches.sort_values(by='유사도', ascending=False).reset_index(drop=True)
        
        st.dataframe(df_matches_sorted, use_container_width=True)
        
        # CSV 다운로드 버튼
        csv_data = to_csv(df_matches_sorted)
        st.download_button(
            label="매칭 결과 CSV 다운로드",
            data=csv_data,
            file_name='project_alignment_matches.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
