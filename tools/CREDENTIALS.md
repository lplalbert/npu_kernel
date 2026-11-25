# CREDENTIALS · 凭据回填说明

> 公开仓库，**禁止**把真实凭据写进任何提交的文件。
> 所有 tools/ 脚本一律从环境变量读取凭据，运行前先 export：

```bash
# cannjudge.cn 判题平台（三个判题赛通用：天梯赛 / mfg / S10）
export CANNJUDGE_JWT='<cannjudge_auth 的 Cookie 值>'
export CANNJUDGE_USER='<你的 userId>'
```

## 如何获取

1. 浏览器登录 `https://cannjudge.cn`（需要能访问判题平台网络）
2. F12 → Application → Cookies → `https://cannjudge.cn` → 复制 `cannjudge_auth` 的整串值 → 就是 `CANNJUDGE_JWT`
3. userId：同上页面的登录态里有；也可以直接解 JWT payload 的 `uid` 字段：
   ```bash
   echo '<JWT前半段>' | cut -d. -f2 | base64 -d 2>/dev/null
   ```

## 注意事项

- JWT 有有效期（历史 token 约 1~2 周）。过期后按上面步骤重取即可，**不要**把新 token 提交进仓库。
- 提交/查分脚本清单：`ladder_submit.py`、`mfg_submit3.py`、`subpoll.py`、`sub_zip.py`、`try_submit.py`、
  `fetch_subs.py`、`fetch_all_scores.py`、`check_quota.py`、`one_detail.py`、`poll_all.py`、`find_my_subs.py`、
  `get_progress.py`、`exact_prec.py`。
- 旧开发机 SSH（`plink` / hostkey / 内网 IP）不在此文件管理；新机器连接信息请放在**本地**（不进仓库）。
