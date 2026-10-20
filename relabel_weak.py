"""
弱类样本 LLM 重打标(flash 模型)
用途: 对 test 报告中的弱类别(大部件/变流系统类/润滑系统类/其他/液压件)样本,
      找出「标注 vs 当前模型预测不一致」的可疑样本,调用 flash 模型重新打标供人工复核。

用法:
  python3 relabel_weak.py            # 无 key: 只导出可疑清单(dry-run)
  FLASH_API_KEY=xxx python3 relabel_weak.py   # 有 key: 调用 API 重打标

API 配置(环境变量):
  FLASH_API_KEY   必填(重打标时)          例: 智谱/DeepSeek 的 key
  FLASH_BASE_URL  默认 https://api.z.ai/api/paas/v4
  FLASH_MODEL     默认 glm-4.6-flash      (按你的账号可用型号调整)
"""
import json, os, sys, time
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
WEAK = {
    'type': {'大部件'},
    'scene': {'变流系统类', '润滑系统类', '控制系统类'},
    'physical': {'其他', '液压件', '密封件'},
}
SYSTEM_PROMPT = """你是风电备件分类专家。根据物料名称从三个维度分类,严格按JSON数组返回 [{name, type, physical, scene}]。
1. type: 大部件(整机/叶片/齿轮箱整机/发电机整机/塔筒等) / 工器具 / 耗材(定检消耗品) / 备件(替换件)
2. physical: 电气件 / 机械件 / 电子件 / 传感器 / 油品 / 液压件 / 密封件 / 其他
3. scene: 偏航系统类 / 变桨系统类 / 齿轮箱类 / 发电机类 / 液压系统类 / 润滑系统类 / 控制系统类 / 变流系统类 / 通用类
规则: 名称明确含系统关键词才归该系统,否则通用类;润滑油/脂等成品油 physical=油品、scene=通用类。"""

def collect_suspects():
    """弱类中 标注与模型预测不一致 的样本 = 最可疑的标签噪声"""
    from classifier import WindClassifier
    clf = WindClassifier()
    suspects = []
    for split in ['train', 'val']:
        with open(os.path.join(BASE, 'data', f'{split}.json')) as f:
            rows = json.load(f)
        for it in rows:
            weak_hit = any(it[d] in labels for d, labels in WEAK.items())
            if not weak_hit:
                continue
            r = clf.classify(it['name'])
            if any(r[d] != it[d] for d in ['type', 'physical', 'scene']):
                suspects.append({'split': split, 'name': it['name'],
                                 'old': [it['type'], it['physical'], it['scene']],
                                 'model': [r['type'], r['physical'], r['scene']]})
    return suspects

def call_flash(batch_names, key, base_url, model):
    lines = '\n'.join(f'{i}. {n}' for i, n in enumerate(batch_names, 1))
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': '请分类以下物料,返回JSON数组:\n\n' + lines},
        ],
        'max_tokens': 4000, 'temperature': 0.1,
    }
    for retry in range(3):
        try:
            resp = requests.post(f'{base_url}/chat/completions',
                                 headers={'Authorization': f'Bearer {key}'},
                                 json=payload, timeout=60)
            if resp.status_code == 200:
                text = resp.json()['choices'][0]['message']['content']
                s, e = text.find('['), text.rfind(']')
                return json.loads(text[s:e+1]) if s != -1 else None
            print(f'  API {resp.status_code}: {resp.text[:120]}')
        except Exception as ex:
            print(f'  异常: {ex}')
        time.sleep(2 ** retry)
    return None

def main():
    suspects = collect_suspects()
    print(f'可疑样本(弱类且标注≠模型预测): {len(suspects)} 条')
    out_path = os.path.join(BASE, 'data', 'relabel_suggestions.json')
    key = os.environ.get('FLASH_API_KEY')
    if not key:
        with open(out_path, 'w') as f:
            json.dump(suspects, f, ensure_ascii=False, indent=1)
        print(f'dry-run: 清单已导出 {out_path},复核后手动修正或配置 FLASH_API_KEY 重跑')
        return
    base_url = os.environ.get('FLASH_BASE_URL', 'https://api.z.ai/api/paas/v4')
    model = os.environ.get('FLASH_MODEL', 'glm-4.6-flash')
    BATCH = 50
    for i in range(0, len(suspects), BATCH):
        chunk = suspects[i:i+BATCH]
        res = call_flash([s['name'] for s in chunk], key, base_url, model)
        if res and len(res) == len(chunk):
            for s, r in zip(chunk, res):
                s['flash'] = [r.get('type'), r.get('physical'), r.get('scene')]
            print(f'  批 {i//BATCH+1}: {len(chunk)} 条完成')
        else:
            print(f'  批 {i//BATCH+1}: 失败,跳过')
        with open(out_path, 'w') as f:
            json.dump(suspects, f, ensure_ascii=False, indent=1)
    n_flash = sum(1 for s in suspects if 'flash' in s)
    print(f'完成: {n_flash} 条已重打标 → {out_path}')
    print('注意: 建议人工复核 flash 与 old 一致率后再决定是否替换标签并重训')

if __name__ == '__main__':
    main()
