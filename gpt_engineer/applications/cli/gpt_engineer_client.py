"""
GPT Engineer Client Module

This module provides a client class for using gpt-engineer functionality in other projects.
It encapsulates the core functionality of gpt-engineer and provides a simple interface for
generating and improving code.

Example usage:
    from gpt_engineer_client import GPTEngineerClient
    
    client = GPTEngineerClient(
        project_path="./my_project",
        openai_api_key="your-api-key",
        openai_api_base="https://your-api-endpoint"
    )
    
    # Generate code
    comparison_result = client.generate("Create a simple web server")
    
    # Or improve existing code
    comparison_result = client.improve("Optimize performance")
"""

import difflib
from pathlib import Path
from typing import Dict, List, Optional

from gpt_engineer.core.ai import AI
from gpt_engineer.core.default.disk_memory import DiskMemory
from gpt_engineer.core.default.disk_execution_env import DiskExecutionEnv
from gpt_engineer.applications.cli.cli_agent import CliAgent
from gpt_engineer.applications.cli.file_selector import FileSelector
from gpt_engineer.core.default.file_store import FileStore
from gpt_engineer.core.default.paths import PREPROMPTS_PATH, memory_path
from gpt_engineer.core.default.steps import (
    execute_entrypoint,
    gen_code,
    handle_improve_mode,
    improve_fn as improve_fn,
)
from gpt_engineer.core.files_dict import FilesDict
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt
from gpt_engineer.tools.custom_steps import clarified_gen, lite_gen, self_heal
from gpt_engineer.core.git import stage_uncommitted_to_git


class GPTEngineerClient:
    """
    A client class for using gpt-engineer functionality in other projects.
    
    This class encapsulates the core functionality of gpt-engineer and provides
    a simple interface for generating and improving code.
    
    Attributes
    ----------
    project_path : str
        The path to the project directory.
    openai_api_key : str
        The OpenAI API key.
    openai_api_base : str
        The OpenAI API base URL.
    model : str
        The model to use for code generation.
    temperature : float
        The temperature to use for code generation.
    memory : DiskMemory
        The memory object for storing and retrieving information.
    execution_env : DiskExecutionEnv
        The execution environment for running code.
    agent : CliAgent
        The agent for generating and improving code.
    """
    
    def __init__(
        self,
        project_path: str = ".",
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        azure_endpoint: str = "",
        self_heal_mode: bool = False,
        use_custom_preprompts: bool = False,
        verbose: bool = False,
        debug: bool = False,
        use_cache: bool = False,
        clarify_mode: bool = False,
        lite_mode: bool = False,
        diff_timeout: int = 3,
        ignore_files: List[str] = ["example.txt", "new_file.txt"],
    ):
        """
        Initialize the GPTEngineerClient.
        
        Parameters
        ----------
        project_path : str, optional
            The path to the project directory. Default is current directory.
        openai_api_key : str, optional
            The OpenAI API key. If not provided, will try to use environment variable.
        openai_api_base : str, optional
            The OpenAI API base URL. If not provided, will use the default OpenAI API.
        model : str, optional
            The model to use for code generation. Default is "gpt-4o".
        temperature : float, optional
            The temperature to use for code generation. Default is 0.1.
        azure_endpoint : str, optional
            The Azure endpoint to use for code generation. Default is "".
        use_custom_preprompts : bool, optional
            Whether to use custom preprompts. Default is False.
        verbose : bool, optional
            Whether to enable verbose logging. Default is False.
        debug : bool, optional
            Whether to enable debug mode. Default is False.
        use_cache : bool, optional
            Whether to use cache for LLM responses. Default is False.
        diff_timeout : int, optional
            Timeout for diff regexp search. Default is 3.
        """
        self.project_path = project_path
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.model = model
        self.temperature = temperature
        self.azure_endpoint = azure_endpoint
        self.use_custom_preprompts = use_custom_preprompts
        self.verbose = verbose
        self.debug = debug
        self.use_cache = use_cache
        self.diff_timeout = diff_timeout
        self.self_heal_mode = self_heal_mode
        self.clarify_mode = clarify_mode
        self.lite_mode = lite_mode
        self.ignore_files = ignore_files
        
        # Initialize components
        self._initialize_components()
    
    
    def _initialize_components(self):
        """Initialize the components needed for code generation and improvement."""
        # Create AI instance with the provided API key and base URL
        self.ai = AI(
            openai_api_base=self.openai_api_base,
            openai_api_key=self.openai_api_key,
            model_name=self.model,
            temperature=self.temperature,
            azure_endpoint=self.azure_endpoint,
        )
        
        # Set up memory and execution environment
        self.memory = DiskMemory(memory_path(self.project_path))
        self.memory.archive_logs()
        self.execution_env = DiskExecutionEnv()
        
        # Set up preprompts
        preprompts_path = self._get_preprompts_path()
        self.preprompts_holder = PrepromptsHolder(preprompts_path)

        # Configure code generation function
        code_gen_fn = gen_code
        if self.clarify_mode:
            code_gen_fn = clarified_gen
        elif self.lite_mode:
            code_gen_fn = lite_gen
        
        # Configure execution function
        execution_fn = execute_entrypoint
        if self.self_heal_mode:
            execution_fn = self_heal
        
        # Create agent
        self.agent = CliAgent.with_default_config(
            memory=self.memory,
            execution_env=self.execution_env,
            ai=self.ai,
            preprompts_holder=self.preprompts_holder,
            diff_timeout=self.diff_timeout,
            process_code_fn=execution_fn,
            code_gen_fn=code_gen_fn,
            improve_fn=improve_fn,
            ignore_files=self.ignore_files,
        )
    
    def _get_preprompts_path(self) -> Path:
        """
        Get the path to the preprompts, using custom ones if specified.
        
        Returns
        -------
        Path
            The path to the directory containing the preprompts.
        """
        if not self.use_custom_preprompts:
            return PREPROMPTS_PATH
        
        custom_preprompts_path = Path(self.project_path) / "preprompts"
        if not custom_preprompts_path.exists():
            # Copy default preprompts to custom location
            import shutil
            custom_preprompts_path.mkdir(parents=True, exist_ok=True)
            for file in PREPROMPTS_PATH.glob("*"):
                if file.is_file():
                    shutil.copy(file, custom_preprompts_path / file.name)
        
        return custom_preprompts_path
    
    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_file: str = "prompt",
        no_execution: bool = False,
        ignore_files: List[str] = ["example.txt", "new_file.txt"],
    ) -> Dict[str, List[str]]:
        """
        Generate code based on the provided prompt or prompt file and return the comparison result.
        
        Parameters
        ----------
        prompt : str, optional
            The prompt describing what code to generate. If None, will try to read from prompt_file.
        prompt_file : str, optional
            The name of the file containing the prompt. Default is "prompt".
        lite_mode : bool, optional
            Whether to use lite mode for generation. Default is False.
        clarify_mode : bool, optional
            Whether to clarify the prompt with the AI before generation. Default is False.
        self_heal_mode : bool, optional
            Whether to enable self-healing mode. Default is False.
        no_execution : bool, optional
            Whether to skip execution of the generated code. Default is False.
        
        Returns
        -------
        Dict[str, List[str]]
            A dictionary mapping file paths to lists of diff lines.
        """
        # Get prompt from file or parameter
        prompt_str = prompt
        if prompt_str is None:
            # Try to read from file
            import os
            if os.path.isdir(prompt_file):
                raise ValueError(
                    f"The path to the prompt, {prompt_file}, already exists as a directory. "
                    f"No prompt can be read from it. Please specify a valid prompt file."
                )
            
            prompt_str = self.memory.get(prompt_file)
            if prompt_str:
                print(f"Using prompt from file: {prompt_file}")
                print(prompt_str)
            else:
                raise ValueError(
                    f"No prompt provided and could not read from file {prompt_file}. "
                    f"Please provide a prompt or create a prompt file."
                )
    
        # Original files (empty for generation)
        original_files = FilesDict()
        
        # Generate code
        if no_execution:
            # Only generate code without execution
            generated_files = code_gen_fn(
                self.ai, Prompt(prompt_str), self.memory, self.preprompts_holder
            )
        else:
            # Generate code and execute
            generated_files = self.agent.init(Prompt(prompt_str), diff_timeout=self.diff_timeout)
        
        # 自动应用更改
        if generated_files is not None and len(generated_files) > 0:
            process_files = generated_files.copy()
            for file_path in generated_files.keys():
                for ignore_file in self.ignore_files:
                    if ignore_file in file_path:
                        process_files.pop(file_path)
                        break
            stage_uncommitted_to_git(Path(self.project_path), process_files, True)
            file_store = FileStore(self.project_path)
            file_store.push(process_files)
        
        # Compare original (empty) with generated files
        return self.compare(original_files, process_files)
    
    def improve(
        self,
        prompt: Optional[str] = None,
        prompt_file: str = "prompt",
        files_dict: Optional[FilesDict] = None,
        skip_file_selection: bool = False,
        no_execution: bool = False,
        diff_timeout: int = 3,
        ignore_files: List[str] = [],
    ) -> Dict[str, List[str]]:
        """
        Improve existing code based on the provided prompt or prompt file and return the comparison result.
        
        Parameters
        ----------
        prompt : str, optional
            The prompt describing how to improve the code. If None, will try to read from prompt_file.
        prompt_file : str, optional
            The name of the file containing the prompt. Default is "prompt".
        files_dict : FilesDict, optional
            The files to improve. If not provided, will select files from the project directory.
        skip_file_selection : bool, optional
            Whether to skip interactive file selection. Default is False.
        no_execution : bool, optional
            Whether to skip execution of the improved code. Default is False.
        
        Returns
        -------
        Dict[str, List[str]]
            A dictionary mapping file paths to lists of diff lines.
        """
        # Get prompt from file or parameter
        prompt_str = prompt
        if prompt_str is None:
            # Try to read from file
            import os
            if os.path.isdir(prompt_file):
                raise ValueError(
                    f"The path to the prompt, {prompt_file}, already exists as a directory. "
                    f"No prompt can be read from it. Please specify a valid prompt file."
                )
            
            prompt_str = self.memory.get(prompt_file)
            if prompt_str:
                print(f"Using prompt from file: {prompt_file}")
                print(prompt_str)
            else:
                raise ValueError(
                    f"No prompt provided and could not read from file {prompt_file}. "
                    f"Please provide a prompt or create a prompt file."
                )
        
        # Get original files
        original_files = FilesDict()
        file_store = FileStore(self.project_path)
        is_linting = False
        if files_dict is None:
            # Select files to improve
            original_files, is_linting = FileSelector(self.project_path).ask_for_files(
                skip_file_selection=skip_file_selection
            )
        else:
            original_files = FilesDict(files_dict)
            is_linting = True  # 当提供files_dict时默认启用linting
        
        # Make a deep copy of the original files
        import copy
        files_to_improve = copy.deepcopy(original_files)
        
        # Improve code
        if no_execution:
            # Only improve code without execution
            improved_files = self.agent.improve_fn(
                self.ai,
                Prompt(prompt_str),
                files_to_improve,
                self.memory,
                self.preprompts_holder,
                diff_timeout=self.diff_timeout,
                ignore_files=self.ignore_files,
            )
        else:
            # Improve code and handle execution
            improved_files = handle_improve_mode(
                Prompt(prompt_str), self.agent, self.memory, files_to_improve, diff_timeout=self.diff_timeout, ignore_files=self.ignore_files
            )
        
        # 在应用更改前对改进后的代码进行linting
        if is_linting:
            print("\nApplying linting to improved files...\n")
            improved_files = file_store.linting(improved_files)
        
        # 自动应用更改
        if improved_files is not None and len(improved_files) > 0:
            process_files = improved_files.copy()
            for file_path in improved_files.keys():
                for ignore_file in self.ignore_files:
                    if ignore_file in file_path:
                        process_files.pop(file_path)
                        break
            stage_uncommitted_to_git(Path(self.project_path), process_files, True)
            file_store.push(process_files)
        
        # Compare original with improved files
        return self.compare(original_files, process_files)
    
    def compare(self, f1: FilesDict, f2: FilesDict) -> Dict[str, str]:
        """
        Compare two file dictionaries and return the differences in Markdown format.
        
        Parameters
        ----------
        f1 : FilesDict
            The first file dictionary.
        f2 : FilesDict
            The second file dictionary.
        
        Returns
        -------
        Dict[str, str]
            A dictionary mapping file paths to Markdown-formatted diff strings.
        """
        result = {}
        
        # Find all files in either dictionary
        all_files = sorted(set(f1) | set(f2))
        
        for file_path in all_files:
            # Handle files that exist in only one dictionary
            if file_path not in f1:
                result[file_path] = f"**New File:** `{file_path}`\n\n```\n{f2[file_path]}\n```"
                continue
            
            if file_path not in f2:
                result[file_path] = f"**Deleted File:** `{file_path}`\n\n```\n{f1[file_path]}\n```"
                continue
            
            # Skip if files are identical
            if f1[file_path] == f2[file_path]:
                continue
            
            # Generate diff in Markdown format
            lines1 = f1[file_path].splitlines()
            lines2 = f2[file_path].splitlines()
            
            diff_lines = list(difflib.unified_diff(
                lines1,
                lines2,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            ))
            
            # Convert diff to Markdown with syntax highlighting
            md_diff = []
            md_diff.append(f"**Changes to:** `{file_path}`\n")
            md_diff.append("```diff")
            
            for line in diff_lines:
                md_diff.append(line)
            
            md_diff.append("```")
            
            result[file_path] = "\n".join(md_diff)
        
        return result