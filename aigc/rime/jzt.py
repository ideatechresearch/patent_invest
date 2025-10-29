from collections import deque
from rime.allele import Allele


def stream_print_genotypes(gen_iter, max_batches=9, start_batch=8):
    """
    Stream printing:
      - 第1批：8 个（每行 2 个，4 行）
      - 第2批：16 个（每行 4 个，4 行）
      - 第3批：32 个（每行 8 个，4 行）
      ...
    这里的阈值是累计阈值（cumulative），确保每次打印的是“下一批的增量”。
    """
    buffer = deque()
    # 生成增量批次大小：8,16,32,...
    batch_sizes = [start_batch * (i + 1) for i in range(max_batches)]
    # 生成累计阈值：8, 8+16=24, 24+32=56, ...
    thresholds = []
    cum = 0
    for sz in batch_sizes:
        cum += sz
        thresholds.append(cum)

    current_idx = 0
    for count, item in enumerate(gen_iter):
        buffer.append(item)

        # 触发所有已经达到的累计阈值（一般只会触发一次）
        while current_idx < len(thresholds) and count >= thresholds[current_idx]:
            # 这个批次要打印的条目数（增量大小）
            batch_size = batch_sizes[current_idx]  # 注意：用增量大小作为本批打印量
            cols = max(1, batch_size // 4)  # 每批 4 行
            print(f"\n=== 第 {current_idx + 1} 批 ({batch_size} 个, 每行 {cols} 个) ===")
            # 逐行从 buffer 中弹出并打印 batch_size 个元素
            for _ in range(4):
                row_items = []
                for _ in range(cols):
                    if not buffer:
                        break
                    row_items.append(buffer.popleft())
                # 如果某行没有足够元素，也按已有元素打印（保持稳定）
                print("  ".join(map(str, row_items)))
            current_idx += 1

    # 最后若有剩余，按合适列宽打印
    if buffer:
        remaining = len(buffer)
        batch_size = batch_sizes[current_idx]
        if current_idx > 0:
            last_cols = max(1, batch_size // 4)
        else:
            last_cols = min(8, max(1, start_batch // 4))
        print(
            f"\n\n=== 第 {current_idx + 1} 批 ({batch_size} 个, 每行 {last_cols} 个), 剩余 {remaining} 个未满一完整批 ===")
        row = []
        while buffer:
            row.append(buffer.popleft())
            if len(row) >= last_cols:
                print("  ".join(map(str, row)))
                row = []
        if row:
            print("  ".join(map(str, row)))


if __name__ == '__main__':
    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
    stream_print_genotypes(genotypes_iter)
