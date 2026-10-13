import sys
import pytest

def main():
    print("Running complete test suite with pytest...")
    ret = pytest.main(["tests/", "-v", "--tb=short"])
    print(f"\nPytest Exit Code: {ret}")
    sys.exit(ret)

if __name__ == "__main__":
    main()
