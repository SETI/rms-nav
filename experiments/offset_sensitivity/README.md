# Offset Sensitivity Analysis

## Example Command Lines:

```bash
python experiments/offset_sensitivity/generate_offset_tasks.py --model-template my_model.json --output tasks.json --u-min -1 --u-max 1 --u-stride 0.25 --v-min -1 --v-max 1 --v-stride 0.01
python main/nav_offset_cloud_tasks.py --task-file tasks.json --nav-results-root offset_results --num-cpus 16
python experiments/offset_sensitivity/analyze_offset_results.py --nav-results-root offset_results
```
