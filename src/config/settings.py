import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    acled_api_key: str | None = os.getenv("ACLED_API_KEY")
    acled_email: str | None = os.getenv("ACLED_EMAIL")

    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "hotspot")
    postgres_user: str = os.getenv("POSTGRES_USER", "hotspot")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "hotspot")

    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic_acled: str = os.getenv("KAFKA_TOPIC_ACLED", "acled-events-nigeria")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
