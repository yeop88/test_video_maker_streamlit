"""영상을 이해해 하이라이트를 고르고, AI 나레이션과 한글 자막을 입혀 쇼츠/릴스로 만드는 패키지."""
import truststore

# 버전은 여기 한 곳에서만 관리한다 (Semantic Versioning, CHANGELOG.md와 Git 태그 vX.Y.Z가 같아야 함)
__version__ = "0.4.0"

# 사내 프록시·보안 프로그램이 HTTPS를 가로채는 환경에서도 모델 다운로드가 되도록
# Python 기본 인증서 묶음 대신 OS 인증서 저장소를 사용한다.
truststore.inject_into_ssl()
