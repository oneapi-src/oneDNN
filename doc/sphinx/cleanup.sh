for f in rst/*.rst; do mv "$f" "$(echo "$f" | sed s/page_dev_guide/dev_guide/)"; done
for f in rst/group_Dnnl*.rst; do mv "$f" "$(echo "$f" | sed s/group_Dnnl/grop_dnnl/)"; done
for f in rst/grop_dnnl*.rst; do mv "$f" "$(echo "$f" | sed s/grop_dnnl/group_dnnl/)"; done
