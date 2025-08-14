import yaml
import json


def yaml_to_dict(yaml_file_path):
    """
    Convert a YAML file to a dictionary.

    Args:
        yaml_file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary representation of the YAML file content.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            yaml_dict = yaml.safe_load(file)
            return yaml_dict
    except FileNotFoundError:
        print(f"File not found: {yaml_file_path}")
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}")


def read_json_list(file_path):
    """
    Reads a list of JSON objects from a file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    list: A list of dictionaries containing the JSON objects.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def evaluate_function_call(tool_definition, function_call):
    # Map for type conversion
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }

    # Check if the function name matches
    if tool_definition["name"] != function_call["name"]:
        return False, "Function name does not match."

    # Check if all required parameters are present
    required_params = [key for key, value in tool_definition["parameter"]
                       ["properties"].items() if value.get("required", False)]
    missing_params = [
        param for param in required_params if param not in function_call["arguments"]]
    if missing_params:
        return False, f"Missing required parameters: {missing_params}"

    # Check if all provided parameters are valid and their data types are correct
    valid_params = tool_definition["parameter"]["properties"]
    invalid_params = []
    type_mismatches = []

    for param, value in function_call["arguments"].items():
        if param not in valid_params:
            invalid_params.append(param)
        else:
            expected_type = valid_params[param]["type"]
            if expected_type not in type_map:
                return False, f"Unsupported parameter type: {expected_type}"
            if not isinstance(value, type_map[expected_type]):
                type_mismatches.append(
                    (param, expected_type, type(value).__name__))

    if invalid_params:
        return False, f"Invalid parameters provided: {invalid_params}"

    if type_mismatches:
        return False, f"Type mismatches: {type_mismatches}"

    return True, "Function call is valid."

def evaluate_function_call_from_toolbox(toolbox, function_call):
    tool_name = function_call["name"]
    this_tool_dec = toolbox.get_one_tool_by_one_name(tool_name)
    if this_tool_dec is None:
        return False, "Tool not found."
    results, results_message = evaluate_function_call(this_tool_dec, function_call)
    return results, results_message
    

def compare_function_calls(pred_function_call, gt_function_call, compare_arguments=True, compare_value=True):
    # Extracting the name and arguments from the predicted function call
    pred_name = pred_function_call["name"]
    pred_arguments = pred_function_call["arguments"]

    # Extracting the name and arguments from the ground truth function call
    gt_name = gt_function_call["name"]
    gt_arguments = gt_function_call["arguments"]

    # Compare function names
    if pred_name != gt_name:
        return False, "Function names do not match."

    if compare_arguments:
        # Compare arguments
        if set(pred_arguments.keys()) != set(gt_arguments.keys()):
            missing_in_pred = set(gt_arguments.keys()) - set(pred_arguments.keys())
            missing_in_gt = set(pred_arguments.keys()) - set(gt_arguments.keys())
            return False, f"Argument keys do not match. Missing in predicted: {missing_in_pred}, Missing in ground truth: {missing_in_gt}"
    if compare_value:
        # Compare argument values
        mismatched_values = []
        for key in pred_arguments:
            if pred_arguments[key] != gt_arguments[key]:
                mismatched_values.append((key, pred_arguments[key], gt_arguments[key]))

        if mismatched_values:
            return False, f"Argument values do not match: {mismatched_values}"

    return True, "Function calls match."


def extract_function_call_json(lst, return_message=False, verbose=True):
    if type(lst) is dict:
        if return_message:
            return lst, ""
        return lst
    result_str = ''.join(lst)
    if verbose:
        print("\033[1;34mPossible LLM outputs for function call:\033[0m", result_str)
    try:
        function_call_json = json.loads(result_str.strip())
        if return_message:
            return function_call_json, ""
        return function_call_json
    except json.JSONDecodeError:
        try:
            index_start = result_str.find(
                '[TOOL_CALLS]') 
            index_end = result_str.find('</s>')
            if index_end == -1:
                index_end = result_str.find('<|eom_id|>')
            if index_end == -1:
                function_call_str = result_str[index_start+ len('[TOOL_CALLS]'):]
            else:
                function_call_str = result_str[index_start+ len('[TOOL_CALLS]'):index_end]
            # print("function_call_str", function_call_str)
            function_call_json = json.loads(function_call_str.strip())
            if return_message:
                message = result_str[:index_start]
                return function_call_json, message
            return function_call_json
        except json.JSONDecodeError:
            try:
                print("Multiple function calls not implemented for 'llama' format.")
                index_start = result_str.find(
                    '<functioncall>') + len('<functioncall>')
                index_end = result_str.find('</functioncall>')
                function_call_str = result_str[index_start:index_end]
                # function_call_str = function_call_str.replace("'", '"')
                function_call_json = json.loads(function_call_str.strip())
                return function_call_json
            except json.JSONDecodeError as e:
                print("Not a function call:", e)
                if return_message:
                    return None, result_str
                return None


def extract_function_call_json_from_qwen(lst, return_message=False, verbose=True):
    """
    专门处理Qwen格式的工具调用提取函数
    
    Qwen格式示例:
    <think>思考内容</think>
    <tool_call>{"name": "Tool_RAG", "arguments": {"description": "...", "limit": 1}}</tool_call>
    
    Args:
        lst: 输入列表或字符串
        return_message: 是否返回消息内容
        verbose: 是否打印调试信息
        
    Returns:
        tuple: (function_call_json, message) 或 function_call_json
    """
    if type(lst) is dict:
        if return_message:
            return lst, ""
        return lst
    
    # 合并列表为字符串
    result_str = ''.join(lst)
    
    if verbose:
        print("\033[1;34mQwen LLM outputs for function call:\033[0m", result_str)
    
    try:
        # 1. 尝试直接解析JSON（如果整个字符串就是JSON）
        function_call_json = json.loads(result_str.strip())
        if return_message:
            return function_call_json, ""
        return function_call_json
    except json.JSONDecodeError:
        pass
    
    try:
        # 2. 尝试提取 <tool_call> 格式
        tool_call_start = result_str.find('<tool_call>')
        tool_call_end = result_str.find('</tool_call>')
        
        if tool_call_start != -1 and tool_call_end != -1:
            # 提取tool_call标签内的JSON内容
            json_start = tool_call_start + len('<tool_call>')
            function_call_str = result_str[json_start:tool_call_end].strip()
            
            # 解析JSON
            function_call_json = json.loads(function_call_str)
            
            if return_message:
                # 提取think部分作为消息
                think_start = result_str.find('<think>')
                think_end = result_str.find('</think>')
                if think_start != -1 and think_end != -1:
                    think_start += len('<think>')
                    message = result_str[think_start:think_end].strip()
                else:
                    # 如果没有think标签，返回tool_call之前的内容
                    message = result_str[:tool_call_start].strip()
                
                return function_call_json, message
            
            return function_call_json
            
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse JSON in <tool_call>: {e}")
    
    try:
        # 3. 尝试提取 [TOOL_CALLS] 格式（兼容原有格式）
        index_start = result_str.find('[TOOL_CALLS]')
        if index_start != -1:
            index_end = result_str.find('</s>')
            if index_end == -1:
                index_end = result_str.find('<|eom_id|>')
            if index_end == -1:
                function_call_str = result_str[index_start + len('[TOOL_CALLS]'):]
            else:
                function_call_str = result_str[index_start + len('[TOOL_CALLS]'):index_end]
            
            function_call_json = json.loads(function_call_str.strip())
            
            if return_message:
                message = result_str[:index_start]
                return function_call_json, message
            return function_call_json
            
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse JSON in [TOOL_CALLS]: {e}")
    
    try:
        # 4. 尝试提取 <functioncall> 格式（兼容原有格式）
        index_start = result_str.find('<functioncall>')
        if index_start != -1:
            index_start += len('<functioncall>')
            index_end = result_str.find('</functioncall>')
            if index_end != -1:
                function_call_str = result_str[index_start:index_end]
                function_call_json = json.loads(function_call_str.strip())
                
                if return_message:
                    message = result_str[:result_str.find('<functioncall>')]
                    return function_call_json, message
                return function_call_json
                
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse JSON in <functioncall>: {e}")
    
    # 5. 所有格式都失败
    if verbose:
        print("Not a valid function call format for Qwen")
    
    if return_message:
        return None, result_str
    return None


def extract_function_call_json_from_qwen(lst, return_message=False, verbose=True):
    """
    专门处理Qwen格式的工具调用提取函数
    
    Qwen格式示例:
    <think>思考内容</think>
    <tool_call>{"name": "Tool_RAG", "arguments": {"description": "...", "limit": 1}}</tool_call>
    
    Args:
        lst: 输入列表或字符串
        return_message: 是否返回消息内容
        verbose: 是否打印调试信息
        
    Returns:
        tuple: (function_call_json, message) 或 function_call_json
    """
    if type(lst) is dict:
        if return_message:
            return lst, ""
        return lst
    
    # 合并列表为字符串
    result_str = ''.join(lst)
    
    if verbose:
        print("\033[1;34mQwen LLM outputs for function call:\033[0m", result_str)
    
    try:
        # 1. 尝试直接解析JSON（如果整个字符串就是JSON）
        function_call_json = json.loads(result_str.strip())
        if return_message:
            return function_call_json, ""
        return function_call_json
    except json.JSONDecodeError:
        pass
    
    try:
        # 2. 尝试提取 <tool_call> 格式
        tool_call_start = result_str.find('<tool_call>')
        tool_call_end = result_str.find('</tool_call>')
        
        if tool_call_start != -1 and tool_call_end != -1:
            # 提取tool_call标签内的JSON内容
            json_start = tool_call_start + len('<tool_call>')
            function_call_str = result_str[json_start:tool_call_end].strip()
            
            # 解析JSON
            function_call_json = json.loads(function_call_str)
            
            if return_message:
                # 提取think部分作为消息
                think_start = result_str.find('<think>')
                think_end = result_str.find('</think>')
                if think_start != -1 and think_end != -1:
                    think_start += len('<think>')
                    message = result_str[think_start:think_end].strip()
                else:
                    # 如果没有think标签，返回tool_call之前的内容
                    message = result_str[:tool_call_start].strip()
                
                return function_call_json, message
            
            return function_call_json
            
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse JSON in <tool_call>: {e}")
    
    try:
        # 3. 尝试提取 [TOOL_CALLS] 格式（兼容原有格式）
        index_start = result_str.find('[TOOL_CALLS]')
        if index_start != -1:
            index_end = result_str.find('</s>')
            if index_end == -1:
                index_end = result_str.find('<|eom_id|>')
            if index_end == -1:
                function_call_str = result_str[index_start + len('[TOOL_CALLS]'):]
            else:
                function_call_str = result_str[index_start + len('[TOOL_CALLS]'):index_end]
            
            function_call_json = json.loads(function_call_str.strip())
            
            if return_message:
                message = result_str[:index_start]
                return function_call_json, message
            return function_call_json
            
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse JSON in [TOOL_CALLS]: {e}")
    
    try:
        # 4. 尝试提取 <functioncall> 格式（兼容原有格式）
        index_start = result_str.find('<functioncall>')
        if index_start != -1:
            index_start += len('<functioncall>')
            index_end = result_str.find('</functioncall>')
            if index_end != -1:
                function_call_str = result_str[index_start:index_end]
                function_call_json = json.loads(function_call_str.strip())
                
                if return_message:
                    message = result_str[:result_str.find('<functioncall>')]
                    return function_call_json, message
                return function_call_json
                
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Failed to parse JSON in <functioncall>: {e}")
    
    # 5. 所有格式都失败
    if verbose:
        print("Not a valid function call format for Qwen")
    
    if return_message:
        return None, result_str
    return None
