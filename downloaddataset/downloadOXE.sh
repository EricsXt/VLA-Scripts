#!/bin/bash
exec > >(tee -a download.log) 2>&1 # 输出写入 download.log
# 设置基础路径
BASE_DIR="/mnt/data/RawData"
GCS_BASE="gs://gresearch/robotics"

# 数据集列表 - 在这里添加你想要下载的数据集
finished_datasets=(
    "berkeley_rpt_converted_externally_to_rlds"
    "roboturk"

    
    "mimic_play"
    "io_ai_tech"
)
datasets=(

    # "robo_set" 
    # "tidybot"
    # "vima_converted_externally_to_rlds"
    # "spoc"
    # "plex_robosuite"
    #  "aloha_mobile"
    # "berkeley_autolab_ur5"
    # "berkeley_cable_routing"
    # "berkeley_fanuc_manipulation"
    # "berkeley_mvp"
    # "fractal20220817_data"
    # # "bc_z"
    # "bridge"
    "droid"
    # "fmb"
    # "cmu_franka_exploration_dataset"
    # "cmu_play_fusion"
    # "nyu_franka_play_dataset_converted_externally_to_rlds"
    # "cmu_stretch"
    # "columbia_cairlab_pusht_real"
    # "dlr_edan_shared_control"
    # "dlr_sara_grid_clamp"
    # "dlr_sara_pour"
    # "fmb"
    # "furniture_bench"
    # "imperialcollege_sawyer_wrist_cam"
    # "kaist_nonprehensile"
    # "kuka"

    # "roboturk"
    # "stanford_hydra_dataset"
    # "stanford_kuka_multimodal_dataset"
    # "stanford_mask_vit"
    # "stanford_robocook"
    # "tokyo_u_lsmo"

    # "utaustin_mutex"
    # "jaco_play"
    # "taco_play"
    # "ucsd_kitchen_dataset_converted_externally_to_rlds"
    # "berkeley_mvp_converted_externally_to_rlds"
    # "asu_table_top_converted_externally_to_rlds"
    # "kaist_nonprehensile_converted_externally_to_rlds"
)

# 颜色输出函数
print_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

# 检查gsutil是否安装
check_gsutil() {
    if ! command -v gsutil &> /dev/null; then
        print_error "gsutil 未安装，请先安装 Google Cloud SDK"
        exit 1
    fi
}

# 下载单个数据集
download_dataset() {
    local dataset_name=$1
    local dataset_dir="${BASE_DIR}/${dataset_name}"
    local gcs_path="${GCS_BASE}/${dataset_name}"
    
    print_info "开始处理数据集: ${dataset_name}"

    print_info "检查数据集是否存在..."
    if ! gsutil ls "${gcs_path}/" &> /dev/null; then
        print_warning "数据集 ${dataset_name} 不存在，跳过"
        return 1
    fi

    print_info "查找所有版本..."
    local version_lines=$(gsutil ls "${gcs_path}/" | grep -E "/[0-9]+\.[0-9]+(\.[0-9]+)?/$")
    local versions=()
    while IFS= read -r line; do
        # 提取版本号
        if [[ $line =~ ${gcs_path}/([0-9]+\.[0-9]+(\.[0-9]+)?)/$ ]]; then
            versions+=("${BASH_REMATCH[1]}")
        fi
    done <<< "$version_lines"

    local selected_version=""
    if [ ${#versions[@]} -eq 0 ]; then
        print_error "未找到有效的版本号，请检查数据集是否存在"
        exit 1
    else
        # 除非是特定数据集，否则版本排序，选择最大版本
        if [ "$dataset_name" == "bc_z" ]; then
            selected_version="0.1.0"
        else
            # 版本排序，选择最大版本
            IFS=$'\n' sorted_versions=($(sort -V <<<"${versions[*]}"))
            selected_version="${sorted_versions[-1]}"
        fi
    fi
    print_info "versions: ${versions}"
    # 使用动态版本号设置下载目录
    local download_dir="${dataset_dir}/${selected_version}"
    local latest_version="${gcs_path}/${selected_version}/"
    
    print_info "最终选择版本: ${selected_version}"
    print_info "下载目录: ${download_dir}"
    
    # 创建下载目录
    mkdir -p "${download_dir}"

    print_info "下载所有文件..."
    # 下载整个版本目录下的所有文件到指定目录
    gsutil -m cp -r "${latest_version}*" "${download_dir}"

    if [ $? -eq 0 ]; then
        print_info "数据集 ${dataset_name} 下载完成"
        local file_count=$(find "${download_dir}" -type f | wc -l)
        print_info "共下载 ${file_count} 个文件"
        print_info "文件保存在: ${download_dir}"
    else
        print_error "数据集 ${dataset_name} 下载失败"
        return 1
    fi
}

# 主函数
main() {
    print_info "开始批量下载数据集..."
    print_info "目标目录: ${BASE_DIR}"
    
    # 检查依赖
    check_gsutil
    
    # 创建基础目录
    mkdir -p "${BASE_DIR}"
    
    # 统计信息
    local total_datasets=${#datasets[@]}
    local success_count=0
    local failed_datasets=()
    
    print_info "共需要处理 ${total_datasets} 个数据集"
    
    # 遍历数据集列表
    for i in "${!datasets[@]}"; do
        local dataset=${datasets[$i]}
        local current=$((i + 1))
        
        echo ""
        print_info "进度: [${current}/${total_datasets}] 处理数据集: ${dataset}"
        
        if download_dataset "$dataset"; then
            ((success_count++))
        else
            failed_datasets+=("$dataset")
        fi
        
        # 添加短暂延迟，避免请求过于频繁
        sleep 1
    done
    
    # 输出统计结果
    echo ""
    print_info "============== 下载完成 =============="
    print_info "成功下载: ${success_count}/${total_datasets} 个数据集"
    
    if [ ${#failed_datasets[@]} -gt 0 ]; then
        print_warning "失败的数据集:"
        for failed in "${failed_datasets[@]}"; do
            echo "  - $failed"
        done
    fi
    
    print_info "所有文件保存在: ${BASE_DIR}"
}

# 运行主函数
main "$@" 