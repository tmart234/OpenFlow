import sys
import onnx

def set_version(model, new_version):
    if not model.metadata_props:
        model.metadata_props.append(onnx.StringStringEntryProto(key='version', value=new_version))
    else:
        for prop in model.metadata_props:
            if prop.key == 'version':
                prop.value = new_version
                break
        else:
            model.metadata_props.append(onnx.StringStringEntryProto(key='version', value=new_version))
    return model

def get_version(model):
    for prop in model.metadata_props:
        if prop.key == 'version':
            return prop.value
    return None

if __name__ == '__main__':
    action = sys.argv[1].lower()  # 'get' or 'set'
    model_path = sys.argv[2]

    # Load the model
    model = onnx.load(model_path)

    if action == 'set' and len(sys.argv) > 3:
        new_version = sys.argv[3]  
        model = set_version(model, new_version)
        onnx.save(model, model_path)
        print(f"Set model version to: {new_version}")
    elif action == 'get':
        version = get_version(model)
        if version:
            print(f"Current model version: {version}")
        else:
            print("No version found in model metadata.")
    else:
        raise ValueError("Invalid action. Use 'get' or 'set' followed by the model path and optionally the new version.")
