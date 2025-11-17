import re
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MetinOzetleyici:
    
    def __init__(self, dil: str = 'tr'):
        self.dil = dil
        self.stop_words = self._load_stop_words()
        
    def _load_stop_words(self) -> set:
        turkish_stop_words = {
            've', 'ile', 'bir', 'bu', 'da', 'de', 'den', 'için', 'ki', 'mi', 
            'ne', 'var', 'daha', 'çok', 'gibi', 'veya', 'ama', 'fakat', 'ya',
            'ancak', 'hem', 'olarak', 've', 'ise', 'şu', 'bana', 'beni', 'o',
            'ben', 'sen', 'biz', 'siz', 'onlar', 'şey', 'diye', 'çünkü', 'nasıl',
            'neden', 'nerede', 'ne', 'kim', 'hangi', 'kaç', 'her', 'bazı', 'hiç',
            'tüm', 'bütün', 'kimi', 'kendi', 'olan', 'olduğu', 'oldu', 'vardır',
            'yoktur', 'üzere', 'sonra', 'önce', 'şimdi', 'burada', 'orada',
            'nerede', 'böyle', 'şöyle', 'ya da', 'sadece', 'bile', 'herhangi',
            'aynı', 'başka', 'diğer', 'ilk', 'son', 'göre', 'karşı', 'içinde',
            'dışında', 'arasında', 'sırasında', 'ardından', 'dolayı', 'rağmen'
        }
        
        english_stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        return turkish_stop_words if self.dil == 'tr' else english_stop_words
    
    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def sentence_split(self, text: str) -> List[str]:
        text = re.sub(r'([.!?])\s+', r'\1|SPLIT|', text)
        sentences = text.split('|SPLIT|')
        
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def word_tokenize(self, text: str) -> List[str]:
        text = text.lower()
        
        words = re.findall(r'\b[a-züğışöçâî]+\b', text, re.UNICODE)
        
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return words
    
    def calculate_word_frequencies(self, sentences: List[str]) -> Dict[str, float]:
        word_freq = Counter()
        
        for sentence in sentences:
            words = self.word_tokenize(sentence)
            word_freq.update(words)
        
        if word_freq:
            max_freq = max(word_freq.values())
            word_freq = {word: freq / max_freq for word, freq in word_freq.items()}
        
        return dict(word_freq)
    
    def score_sentences_by_frequency(self, sentences: List[str], 
                                    word_freq: Dict[str, float]) -> Dict[int, float]:
        sentence_scores = {}
        
        for idx, sentence in enumerate(sentences):
            words = self.word_tokenize(sentence)
            if not words:
                sentence_scores[idx] = 0.0
                continue
            
            score = sum(word_freq.get(word, 0) for word in words)
            sentence_scores[idx] = score / len(words)
        
        return sentence_scores
    
    def calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        words1 = set(self.word_tokenize(sent1))
        words2 = set(self.word_tokenize(sent2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self.calculate_sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        return similarity_matrix
    
    def textrank_scores(self, similarity_matrix: np.ndarray, 
                       damping: float = 0.85, 
                       iterations: int = 100) -> np.ndarray:
        n = len(similarity_matrix)
        
        scores = np.ones(n) / n
        
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] = similarity_matrix[i] / row_sum
        
        for _ in range(iterations):
            new_scores = np.ones(n) * (1 - damping) / n
            
            for i in range(n):
                for j in range(n):
                    if similarity_matrix[j][i] > 0:
                        new_scores[i] += damping * scores[j] * similarity_matrix[j][i]
            
            if np.allclose(scores, new_scores, atol=1e-6):
                break
            
            scores = new_scores
        
        return scores
    
    def score_sentences_by_position(self, num_sentences: int) -> Dict[int, float]:
        position_scores = {}
        
        for idx in range(num_sentences):
            if idx < num_sentences * 0.2:
                position_scores[idx] = 1.0
            elif idx > num_sentences * 0.9:
                position_scores[idx] = 0.6
            else:
                position_scores[idx] = 0.3
        
        return position_scores
    
    def combine_scores(self, *score_dicts: Dict[int, float], 
                      weights: Optional[List[float]] = None) -> Dict[int, float]:
        if not score_dicts:
            return {}
        
        if weights is None:
            weights = [1.0] * len(score_dicts)
        
        if len(weights) != len(score_dicts):
            raise ValueError("Ağırlık sayısı skor sözlük sayısıyla eşleşmiyor!")
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        combined = defaultdict(float)
        
        for score_dict, weight in zip(score_dicts, weights):
            for idx, score in score_dict.items():
                combined[idx] += score * weight
        
        return dict(combined)
    
    def ozetle(self, text: str, 
               ozet_orani: float = 0.3,
               min_cumle: int = 3,
               max_cumle: int = 10,
               algoritma: str = 'hybrid') -> str:
        text = self.preprocess_text(text)
        
        if not text:
            return ""
        
        sentences = self.sentence_split(text)
        
        if not sentences:
            return text
        
        if len(sentences) <= min_cumle:
            return text
        
        num_summary_sentences = max(
            min_cumle,
            min(max_cumle, int(len(sentences) * ozet_orani))
        )
        
        final_scores = {}
        
        if algoritma in ['frequency', 'hybrid']:
            word_freq = self.calculate_word_frequencies(sentences)
            freq_scores = self.score_sentences_by_frequency(sentences, word_freq)
            
            if algoritma == 'frequency':
                final_scores = freq_scores
        
        if algoritma in ['textrank', 'hybrid']:
            similarity_matrix = self.build_similarity_matrix(sentences)
            textrank_scores = self.textrank_scores(similarity_matrix)
            textrank_dict = {i: score for i, score in enumerate(textrank_scores)}
            
            if algoritma == 'textrank':
                final_scores = textrank_dict
        
        if algoritma in ['position', 'hybrid']:
            position_scores = self.score_sentences_by_position(len(sentences))
            
            if algoritma == 'position':
                final_scores = position_scores
        
        if algoritma == 'hybrid':
            final_scores = self.combine_scores(
                freq_scores,
                textrank_dict,
                position_scores,
                weights=[0.4, 0.4, 0.2]
            )
        
        top_sentence_indices = sorted(
            final_scores.keys(),
            key=lambda x: final_scores[x],
            reverse=True
        )[:num_summary_sentences]
        
        top_sentence_indices.sort()
        
        summary = ' '.join([sentences[i] for i in top_sentence_indices])
        
        return summary
    
    def batch_ozetle(self, texts: List[str], **kwargs) -> List[str]:
        return [self.ozetle(text, **kwargs) for text in texts]
    
    def get_summary_stats(self, original_text: str, summary: str) -> Dict[str, any]:
        original_sentences = self.sentence_split(original_text)
        summary_sentences = self.sentence_split(summary)
        
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        stats = {
            'orijinal_cumle_sayisi': len(original_sentences),
            'ozet_cumle_sayisi': len(summary_sentences),
            'orijinal_kelime_sayisi': original_words,
            'ozet_kelime_sayisi': summary_words,
            'sikistirma_orani': f"{(summary_words / original_words * 100):.1f}%",
            'cumle_azalma_orani': f"{(len(summary_sentences) / len(original_sentences) * 100):.1f}%"
        }
        
        return stats


def demo():
    print("\n" + "="*80)
    print("METİN ÖZETLEME SİSTEMİ")
    print("="*80 + "\n")
    
    turkish_text = """
    Yapay zeka, bilgisayar biliminin en heyecan verici alanlarından biridir. Son yıllarda 
    makine öğrenmesi ve derin öğrenme teknolojilerindeki gelişmeler sayesinde yapay zeka 
    uygulamaları günlük hayatımızın her alanına girdi. Otomobil sürücüsüz araçlardan tutun da 
    sağlık sektöründe hastalık teşhisine, finans sektöründe risk analizinden tutun da 
    eğitim sektöründe kişiselleştirilmiş öğrenmeye kadar birçok alanda kullanılmaktadır. 
    Doğal dil işleme, bilgisayarların insan dilini anlaması ve işlemesi için kullanılan 
    yapay zeka alt alanıdır. Metin özetleme, duygu analizi, makine çevirisi gibi uygulamalar 
    doğal dil işlemenin önemli kullanım alanlarıdır. Gelecekte yapay zeka teknolojilerinin 
    daha da gelişeceği ve hayatımızı daha fazla etkileyeceği öngörülmektedir. Ancak bu 
    gelişmelerle birlikte etik ve güvenlik konuları da önem kazanmaktadır. Yapay zeka 
    sistemlerinin adil, şeffaf ve güvenli olması için çalışmalar devam etmektedir.
    """
    
    english_text = """
    Artificial intelligence is one of the most exciting fields in computer science. Thanks to 
    recent advances in machine learning and deep learning technologies, AI applications have 
    entered every aspect of our daily lives. From self-driving cars to disease diagnosis in 
    healthcare, from risk analysis in finance to personalized learning in education, AI is 
    being used in many areas. Natural language processing is a subfield of artificial 
    intelligence used to enable computers to understand and process human language. Text 
    summarization, sentiment analysis, and machine translation are important application areas 
    of natural language processing. In the future, AI technologies are expected to develop 
    further and impact our lives even more. However, along with these developments, ethical 
    and security issues are also becoming important. Work continues to ensure that AI systems 
    are fair, transparent, and secure.
    """
    
    print("📄 ORİJİNAL METİN (Türkçe):")
    print("-" * 80)
    print(turkish_text.strip())
    
    ozetleyici_tr = MetinOzetleyici(dil='tr')
    ozet_tr = ozetleyici_tr.ozetle(turkish_text, ozet_orani=0.3, algoritma='hybrid')
    
    print("\n\n✨ ÖZET:")
    print("-" * 80)
    print(ozet_tr)
    
    print("\n\n" + "="*80)
    print("="*80 + "\n")
    
    print("📄 ORIGINAL TEXT (English):")
    print("-" * 80)
    print(english_text.strip())
    
    ozetleyici_en = MetinOzetleyici(dil='en')
    ozet_en = ozetleyici_en.ozetle(english_text, ozet_orani=0.3, algoritma='hybrid')
    
    print("\n\n✨ SUMMARY:")
    print("-" * 80)
    print(ozet_en)
    
    print("\n\n" + "="*80)
    print("Sistem hazır! Kendi metninizi özetlemek için:")
    print("  ozetleyici = MetinOzetleyici(dil='tr')")
    print("  ozet = ozetleyici.ozetle(metin)")
    print("="*80 + "\n")


def streamlit_arayuz():
    import streamlit as st
    
    st.set_page_config(
        page_title="Metin Özetleyici",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        
        h1 {
            color: #1f1f1f;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        h3 {
            color: #4a4a4a;
            font-weight: 500;
            margin-top: 2rem;
        }
        
        .stTextArea textarea {
            font-size: 16px;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            padding: 1rem;
        }
        
        .stTextArea textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 1px #6366f1;
        }
        
        .stButton button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 18px;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid #6366f1;
        }
        
        .stRadio > div {
            display: flex;
            gap: 1rem;
            flex-direction: row;
        }
        
        .success-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 14px;
            color: #6c757d;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("# ✨ Metin Özetleyici")
    st.markdown("#### Metninizi yapay zeka ile anında özetleyin")
    st.markdown("---")
    
    if 'ozet' not in st.session_state:
        st.session_state.ozet = ""
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Metninizi Buraya Yazın")
        
        metin = st.text_area(
            label="Metin",
            value="",
            height=400,
            placeholder="Özetlemek istediğiniz metni buraya yapıştırın veya yazın...",
            label_visibility="collapsed"
        )
        
        st.markdown("### ⚙️ Ayarlar")
        
        col_aya1, col_aya2 = st.columns(2)
        
        with col_aya1:
            dil = st.radio(
                "Dil",
                options=['tr', 'en'],
                format_func=lambda x: '🇹🇷 Türkçe' if x == 'tr' else '🇬🇧 English',
                horizontal=True
            )
        
        with col_aya2:
            ozet_orani = st.slider(
                "Özet Uzunluğu",
                min_value=0.1,
                max_value=0.7,
                value=0.3,
                step=0.1,
                format="%.0f%%",
                help="Özetin orijinal metne göre yüzde oranı"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        ozetle_btn = st.button("🚀 Özet Oluştur", use_container_width=True, type="primary")
    
    with col2:
        st.markdown("### ✨ Özet")
        
        if ozetle_btn and metin.strip():
            with st.spinner('Özet oluşturuluyor...'):
                try:
                    ozetleyici = MetinOzetleyici(dil=dil)
                    ozet = ozetleyici.ozetle(metin, ozet_orani=ozet_orani, algoritma='hybrid')
                    stats = ozetleyici.get_summary_stats(metin, ozet)
                    
                    st.session_state.ozet = ozet
                    st.session_state.stats = stats
                    
                except Exception as e:
                    st.error(f"Hata: {str(e)}")
        
        if st.session_state.ozet:
            st.markdown(
                f"""<div class="success-box">
                    {st.session_state.ozet}
                </div>""",
                unsafe_allow_html=True
            )
            
            if st.session_state.stats:
                st.markdown("### 📊 İstatistikler")
                
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.markdown(
                        f"""<div class="stat-card">
                            <div class="stat-number">{st.session_state.stats['orijinal_cumle_sayisi']}</div>
                            <div class="stat-label">Orijinal Cümle</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                
                with stat_col2:
                    st.markdown(
                        f"""<div class="stat-card">
                            <div class="stat-number">{st.session_state.stats['ozet_cumle_sayisi']}</div>
                            <div class="stat-label">Özet Cümle</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                
                with stat_col3:
                    st.markdown(
                        f"""<div class="stat-card">
                            <div class="stat-number">{st.session_state.stats['sikistirma_orani']}</div>
                            <div class="stat-label">Sıkıştırma</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
        else:
            st.info("👈 Soldan bir metin girin ve 'Özet Oluştur' butonuna tıklayın")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6c757d; font-size: 14px; padding: 1rem;'>
            <p><strong>Nasıl Çalışır?</strong></p>
            <p>Bu sistem, TF-IDF, TextRank ve pozisyon bazlı algoritmaların kombinasyonunu kullanarak<br>
            metninizin en önemli cümlelerini belirler ve size özetini sunar.</p>
            <p style='margin-top: 1rem;'>💡 <em>İpucu: En iyi sonuçlar için en az 5-6 cümlelik metinler kullanın</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.markdown("### 📚 Örnek Metinler")
        st.markdown("---")
        
        if st.button("🇹🇷 Türkçe Örnek", use_container_width=True):
            st.session_state.ornek_metin = """Yapay zeka, bilgisayar biliminin en heyecan verici alanlarından biridir. Son yıllarda makine öğrenmesi ve derin öğrenme teknolojilerindeki gelişmeler sayesinde yapay zeka uygulamaları günlük hayatımızın her alanına girdi. Otomobil sürücüsüz araçlardan tutun da sağlık sektöründe hastalık teşhisine, finans sektöründe risk analizinden tutun da eğitim sektöründe kişiselleştirilmiş öğrenmeye kadar birçok alanda kullanılmaktadır. Doğal dil işleme, bilgisayarların insan dilini anlaması ve işlemesi için kullanılan yapay zeka alt alanıdır."""
            st.rerun()
        
        if st.button("🇬🇧 English Example", use_container_width=True):
            st.session_state.ornek_metin = """Artificial intelligence is one of the most exciting fields in computer science. Thanks to recent advances in machine learning and deep learning technologies, AI applications have entered every aspect of our daily lives. From self-driving cars to disease diagnosis in healthcare, AI is being used in many areas. Natural language processing is a subfield of artificial intelligence."""
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ Hakkında")
        st.markdown("""
        **Metin Özetleme Sistemi**
        
        - 🤖 Hibrit AI algoritmaları
        - 🌍 Türkçe & İngilizce destek
        - ⚡ Anında sonuç
        - 🎯 Yüksek doğruluk
        """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--arayuz":
        streamlit_arayuz()
    else:
        demo()
