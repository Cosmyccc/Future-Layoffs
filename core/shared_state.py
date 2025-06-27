class GlobalState:
    g_github_url: str = None
    g_repo_name: str = None

    @classmethod
    def set_github_url(cls, url: str):
        cls.g_github_url = url

    @classmethod
    def set_repo_name(cls, name: str):
        cls.g_repo_name = name

    @classmethod
    def get_github_url(cls) -> str:
        return cls.g_github_url

    @classmethod
    def get_repo_name(cls) -> str:
        return cls.g_repo_name


global_state = GlobalState()
