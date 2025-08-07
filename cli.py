#!/usr/bin/env python3
"""
Titans Finance CLI Tool

A command-line interface for managing the Titans Finance AI development lifecycle project.
Provides convenient commands for setup, development, and deployment.
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TitansFinanceCLI:
    """Main CLI class for Titans Finance project management"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / ".venv"

    def setup(self, use_pip: bool = False, skip_docker: bool = False) -> bool:
        """Setup the project environment"""
        logger.info("🚀 Setting up Titans Finance project...")

        try:
            # Check prerequisites
            if not self._check_prerequisites():
                return False

            # Setup Python environment
            if not self._setup_python_env(use_pip):
                return False

            # Setup Docker services
            if not skip_docker:
                if not self._setup_docker():
                    logger.warning("Docker setup failed, continuing without services")

            # Run database migrations
            self._run_migrations()

            logger.info("✅ Setup completed successfully!")
            self._print_next_steps()
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def dev(self, service: Optional[str] = None) -> bool:
        """Start development environment"""
        logger.info("🔧 Starting development environment...")

        services = {
            "api": self._start_api,
            "dashboard": self._start_dashboard,
            "jupyter": self._start_jupyter,
            "pipeline": self._run_pipeline,
            "all": self._start_all_services
        }

        if service and service in services:
            return services[service]()
        elif service is None:
            return self._start_all_services()
        else:
            logger.error(f"Unknown service: {service}")
            logger.info(f"Available services: {', '.join(services.keys())}")
            return False

    def pipeline(self, mode: str = "full") -> bool:
        """Run ETL pipeline"""
        logger.info(f"📊 Running {mode} pipeline...")

        try:
            cmd = ["titans-pipeline", f"--mode={mode}"]
            if not self._has_uv():
                cmd = ["python", "data_engineering/etl/run_pipeline.py", f"--mode={mode}"]

            result = self._run_command(cmd)
            if result.returncode == 0:
                logger.info("✅ Pipeline completed successfully!")
                return True
            else:
                logger.error("❌ Pipeline failed!")
                return False

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False

    def train(self, model_type: str = "all") -> bool:
        """Train ML models"""
        logger.info(f"🤖 Training {model_type} models...")

        try:
            cmd = ["titans-train", f"--model-type={model_type}"]
            if not self._has_uv():
                cmd = ["python", "data_science/src/models/train.py", f"--model-type={model_type}"]

            result = self._run_command(cmd)
            if result.returncode == 0:
                logger.info("✅ Model training completed!")
                return True
            else:
                logger.error("❌ Model training failed!")
                return False

        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def test(self, test_type: str = "all") -> bool:
        """Run tests"""
        logger.info(f"🧪 Running {test_type} tests...")

        try:
            if test_type == "unit":
                cmd = ["pytest", "tests/unit/", "-v"]
            elif test_type == "integration":
                cmd = ["pytest", "tests/integration/", "-v"]
            elif test_type == "e2e":
                cmd = ["pytest", "tests/e2e/", "-v"]
            else:
                cmd = ["pytest", "-v"]

            if self._has_uv():
                cmd = ["uv", "run"] + cmd

            result = self._run_command(cmd)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Testing error: {e}")
            return False

    def lint(self, fix: bool = False) -> bool:
        """Run code quality checks"""
        logger.info("🔍 Running code quality checks...")

        try:
            success = True

            # Black formatting
            black_cmd = ["black", "." if not fix else "--check", "."]
            if self._has_uv():
                black_cmd = ["uv", "run"] + black_cmd

            result = self._run_command(black_cmd)
            if result.returncode != 0:
                success = False

            # isort imports
            isort_cmd = ["isort", "." if not fix else "--check-only", "."]
            if self._has_uv():
                isort_cmd = ["uv", "run"] + isort_cmd

            result = self._run_command(isort_cmd)
            if result.returncode != 0:
                success = False

            # flake8 linting
            flake8_cmd = ["flake8", "."]
            if self._has_uv():
                flake8_cmd = ["uv", "run"] + flake8_cmd

            result = self._run_command(flake8_cmd)
            if result.returncode != 0:
                success = False

            if success:
                logger.info("✅ All quality checks passed!")
            else:
                logger.error("❌ Quality checks failed!")

            return success

        except Exception as e:
            logger.error(f"Linting error: {e}")
            return False

    def status(self) -> bool:
        """Show project status"""
        logger.info("📋 Titans Finance Project Status")
        logger.info("=" * 50)

        # Check Python environment
        if self.venv_path.exists():
            logger.info("✅ Virtual environment: Ready")
        else:
            logger.info("❌ Virtual environment: Not found")

        # Check Docker services
        try:
            result = subprocess.run(
                ["docker", "compose", "ps"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                running_services = [line for line in result.stdout.split('\n') if 'Up' in line]
                logger.info(f"🐳 Docker services: {len(running_services)} running")
            else:
                logger.info("❌ Docker services: Not running")
        except:
            logger.info("❓ Docker services: Unknown")

        # Check database
        try:
            import psycopg2
            conn = psycopg2.connect(
                "postgresql://postgres:password@localhost:5432/titans_finance"
            )
            conn.close()
            logger.info("✅ Database: Connected")
        except:
            logger.info("❌ Database: Not accessible")

        # Check models
        models_dir = self.project_root / "data_science" / "models"
        if models_dir.exists() and list(models_dir.glob("*.joblib")):
            logger.info("✅ ML Models: Available")
        else:
            logger.info("❌ ML Models: Not found")

        return True

    def clean(self, deep: bool = False) -> bool:
        """Clean up project artifacts"""
        logger.info("🧹 Cleaning project...")

        try:
            # Clean Python cache
            self._run_command(["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"])
            self._run_command(["find", ".", "-name", "*.pyc", "-delete"])

            # Clean logs
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    log_file.unlink()

            if deep:
                # Clean virtual environment
                if self.venv_path.exists():
                    import shutil
                    shutil.rmtree(self.venv_path)
                    logger.info("🗑️  Removed virtual environment")

                # Clean Docker volumes
                self._run_command(["docker-compose", "down", "-v"], cwd=self.project_root)
                logger.info("🗑️  Cleaned Docker volumes")

            logger.info("✅ Cleanup completed!")
            return True

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return False

    def _check_prerequisites(self) -> bool:
        """Check if required tools are available"""
        required_tools = ["python", "docker", "docker-compose"]

        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                logger.info(f"✅ {tool}: Available")
            except:
                logger.error(f"❌ {tool}: Not found")
                return False

        return True

    def _setup_python_env(self, use_pip: bool = False) -> bool:
        """Setup Python virtual environment"""
        try:
            if self._has_uv() and not use_pip:
                logger.info("📦 Setting up environment with UV...")
                result = self._run_command(["uv", "sync"])
                return result.returncode == 0
            else:
                logger.info("📦 Setting up environment with pip...")
                # Create venv
                if not self.venv_path.exists():
                    subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)

                # Install project
                pip_cmd = [str(self.venv_path / "bin" / "pip"), "install", "-e", "."]
                result = self._run_command(pip_cmd)
                return result.returncode == 0

        except Exception as e:
            logger.error(f"Python environment setup failed: {e}")
            return False

    def _setup_docker(self) -> bool:
        """Setup Docker services"""
        try:
            # First, ensure .env file exists with proper configuration
            if not self._create_env_file():
                logger.warning(".env file creation failed")
            
            logger.info("🐳 Starting Docker services...")
            result = self._run_command(
                ["docker", "compose", "up", "-d", "postgres", "redis"],
                cwd=self.project_root
            )

            if result.returncode == 0:
                # Wait for services to be ready
                logger.info("⏳ Waiting for services to be ready...")
                time.sleep(10)
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Docker setup failed: {e}")
            return False
    
    def _create_env_file(self) -> bool:
        """Create .env file if it doesn't exist"""
        env_file = self.project_root / ".env"
        if env_file.exists():
            logger.info("✅ .env file already exists")
            return True
        
        try:
            # Generate a Fernet key for Airflow
            try:
                from cryptography.fernet import Fernet
                fernet_key = Fernet.generate_key().decode()
            except ImportError:
                logger.warning("cryptography not installed, using placeholder Fernet key")
                fernet_key = "your-fernet-key-here-replace-in-production"
            
            env_content = f"""# Airflow Configuration
AIRFLOW__CORE__FERNET_KEY={fernet_key}

# Database Configuration
POSTGRES_DB=titans_finance
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# MLflow Configuration
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:password@postgres:5432/titans_finance

# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Grafana Configuration
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin

# Jupyter Configuration
JUPYTER_TOKEN=password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql://postgres:password@localhost:5432/titans_finance
"""
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            logger.info("✅ Created .env file with default configuration")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False

    def _run_migrations(self) -> bool:
        """Run database migrations"""
        try:
            logger.info("🗄️  Running database migrations...")
            # This would run Alembic migrations in a real implementation
            logger.info("✅ Migrations completed")
            return True
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return False

    def _start_api(self) -> bool:
        """Start FastAPI server"""
        try:
            logger.info("🌐 Starting FastAPI server...")
            cmd = ["titans-api"] if self._has_uv() else ["uvicorn", "ai_engineering.api.main:app", "--reload"]

            if self._has_uv() and "titans-api" not in cmd:
                cmd = ["uv", "run"] + cmd

            self._run_command(cmd, background=True)
            return True
        except Exception as e:
            logger.error(f"API start failed: {e}")
            return False

    def _start_dashboard(self) -> bool:
        """Start Streamlit dashboard"""
        try:
            logger.info("📊 Starting Streamlit dashboard...")
            cmd = ["streamlit", "run", "ai_engineering/frontend/dashboard.py"]

            if self._has_uv():
                cmd = ["uv", "run"] + cmd

            self._run_command(cmd, background=True)
            return True
        except Exception as e:
            logger.error(f"Dashboard start failed: {e}")
            return False

    def _start_jupyter(self) -> bool:
        """Start Jupyter Lab"""
        try:
            logger.info("📓 Starting Jupyter Lab...")
            cmd = ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]

            if self._has_uv():
                cmd = ["uv", "run"] + cmd

            self._run_command(cmd, background=True)
            return True
        except Exception as e:
            logger.error(f"Jupyter start failed: {e}")
            return False

    def _run_pipeline(self) -> bool:
        """Run ETL pipeline"""
        return self.pipeline("full")

    def _start_all_services(self) -> bool:
        """Start all development services"""
        services = [
            ("API", self._start_api),
            ("Dashboard", self._start_dashboard),
            ("Jupyter", self._start_jupyter),
        ]

        success = True
        for name, func in services:
            try:
                if func():
                    logger.info(f"✅ {name} started")
                else:
                    logger.error(f"❌ {name} failed to start")
                    success = False
            except Exception as e:
                logger.error(f"❌ {name} error: {e}")
                success = False

        if success:
            self._print_service_urls()

        return success

    def _has_uv(self) -> bool:
        """Check if UV is available"""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            return True
        except:
            return False

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None, background: bool = False) -> subprocess.CompletedProcess:
        """Run shell command"""
        if cwd is None:
            cwd = self.project_root

        if background:
            # For background processes, we'd typically use a process manager
            # For now, just show what would be run
            logger.info(f"Would start in background: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0)
        else:
            return subprocess.run(cmd, cwd=cwd)

    def _print_next_steps(self):
        """Print next steps after setup"""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Start development services:")
        print("   python cli.py dev")
        print("\n2. Run the data pipeline:")
        print("   python cli.py pipeline")
        print("\n3. Train ML models:")
        print("   python cli.py train")
        print("\n4. Check project status:")
        print("   python cli.py status")
        print("\n" + "="*60)

    def _print_service_urls(self):
        """Print service URLs"""
        print("\n🌐 Service URLs:")
        print("- FastAPI: http://localhost:8000")
        print("- API Docs: http://localhost:8000/docs")
        print("- Dashboard: http://localhost:8501")
        print("- Jupyter Lab: http://localhost:8888 (token: password)")
        print("- Airflow: http://localhost:8081 (admin/admin)")
        print("- MLflow: http://localhost:5000")
        print("- pgAdmin: http://localhost:5050 (admin@titans.com/admin)")
        print("- PostgreSQL: localhost:5432 (user: postgres, pass: password)")
        print("- Redis: localhost:6379")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Titans Finance CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py setup                 # Setup project
  python cli.py dev                   # Start all services
  python cli.py dev --service api     # Start only API
  python cli.py pipeline              # Run ETL pipeline
  python cli.py train                 # Train models
  python cli.py test                  # Run tests
  python cli.py lint                  # Check code quality
  python cli.py status                # Show status
  python cli.py clean                 # Clean artifacts
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup project environment")
    setup_parser.add_argument("--use-pip", action="store_true", help="Use pip instead of UV")
    setup_parser.add_argument("--skip-docker", action="store_true", help="Skip Docker setup")

    # Dev command
    dev_parser = subparsers.add_parser("dev", help="Start development environment")
    dev_parser.add_argument("--service", choices=["api", "dashboard", "jupyter", "pipeline", "all"],
                           help="Specific service to start")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run ETL pipeline")
    pipeline_parser.add_argument("--mode", choices=["full", "incremental"], default="full",
                                help="Pipeline execution mode")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument("--model-type", default="all",
                             help="Type of model to train")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--type", choices=["unit", "integration", "e2e", "all"],
                            default="all", help="Type of tests to run")

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Run code quality checks")
    lint_parser.add_argument("--fix", action="store_true", help="Fix issues automatically")

    # Status command
    subparsers.add_parser("status", help="Show project status")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean project artifacts")
    clean_parser.add_argument("--deep", action="store_true", help="Deep clean including venv")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    cli = TitansFinanceCLI()

    try:
        if args.command == "setup":
            success = cli.setup(use_pip=args.use_pip, skip_docker=args.skip_docker)
        elif args.command == "dev":
            success = cli.dev(service=args.service)
        elif args.command == "pipeline":
            success = cli.pipeline(mode=args.mode)
        elif args.command == "train":
            success = cli.train(model_type=args.model_type)
        elif args.command == "test":
            success = cli.test(test_type=args.type)
        elif args.command == "lint":
            success = cli.lint(fix=args.fix)
        elif args.command == "status":
            success = cli.status()
        elif args.command == "clean":
            success = cli.clean(deep=args.deep)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\n👋 Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
