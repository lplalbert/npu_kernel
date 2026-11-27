# Ascend C 绠楀瓙绔炶禌 路 浜ゆ帴鍖?
> 涓夋潯绾垮苟琛岋細**CANN 绠楀瓙澶╂璧涳紙10-11鏈堣禌锛?* / **鍒堕€犱笟绠楀瓙璧涳紙gcc2026_mfg锛?* / **鏄囪吘绠楀瓙鎸戞垬璧?S10**
> 鐘舵€佸揩鐓ф椂闂达細**2026-11-25**锛堝ぉ姊禌鎴 11-30 17:00锛宮fg 鎴 12-07锛?
---

## 涓€銆佹垚缁╂€昏锛堟埅鑷?2026-11-25锛屽叏閮ㄤ负骞冲彴鐪熷疄鍒ら缁撴灉锛?
| 璧涢亾 | 缁撴灉 | 璇存槑 |
|---|---|---|
| **澶╂璧?* | **14 / 18 棰?Pass** | 涓夐亾 100 涓囩Н鍒嗘。鍏ㄩ儴鎷夸笅锛團lashAttention / ConvTranspose2D / DepthwiseConv2D锛?|
| **鍒堕€犱笟璧?mfg** | **13 / 15 棰?Pass** | 鍓╀綑 2 閬擄細spspmm锛圵A锛夈€乺adius锛圵A锛屽凡鏀惧純锛?|
| **S10 鎸戞垬璧?* | **5 / 5 棰?Pass** | 鍏ㄩ儴 Pass锛宺educe 鍙﹀仛浜?19.5脳 鎻愰€?|

澶╂璧?Pass 鐨?14 閬撹鐩?*鍏ㄩ儴 5 涓毦搴︽。**锛?
| 闅惧害 | 绠楀瓙 | 骞冲彴鎺掑悕 | 寰楀垎 | submissionId |
|---|---|---|---|---|
| 5 | flash_attention | 62 | 10.25 | `6b05c00985ad43bbe0653413` |
| 5 | conv_transpose2_d | 41 | 8.00 | `6b05dd9a85ad43bbe0ab47ae` |
| 5 | depthwise_conv2_d | 24 | 12.89 | `6b0619d285ad43bbe017f9b6` |
| 4 | gf2_poly | 30 | 11.35 | `6b05c8ae85ad43bbe07a78e6` |
| 4 | cube1 | 21 | 6.29 | `6b06272c85ad43bbe02e38d3` |
| 4 | cube2 | 22 | 7.34 | `6b062c7c85ad43bbe037751f` |
| 3 | gf2k_mul_square | 24 | 19.23 | `6b05c96985ad43bbe07c4cd5` |
| 3 | quant_matmul_sigmoid_mul_quant | 32 | 9.35 | `6b05d0d785ad43bbe08e2057` |
| 3 | mhc_sinkhorn_grad | 56 | 0.00 (WA) | `6b05bd3b85ad43bbe05e264f` |
| 2 | bit_plane_transform | 26 | 23.30 | `6b05c5ee85ad43bbe073be10` |
| 2 | group_norm | 24 | 24.43 | `6b05bece85ad43bbe0624322` |
| 2 | mse_loss | 29 | 34.91 | `6b05c20785ad43bbe06a0343` |
| 2 | group_norm_silu_quant | **11** | 16.96 | `6b05f38985ad43bbe0d75fa9` |
| 2 | confusion_mul_grad | 90 | 0.00 (WA) | `6b05be0585ad43bbe0605351` |
| 2 | multi_add_rms_norm_dynamic_quant | 57 | 0.00 (WA) | `6b05c10585ad43bbe067a2a3` |
| 2 | index | 33 | 0.00 (RE) | `6b05c30785ad43bbe06c71c3` |
| 1 | trace | 125 | 18.57 | `6b05caeb85ad43bbe07fecad` |
| 1 | bn_training_update | 41 | 19.60 | `6b06110985ad43bbe009c20d` |

> 瀹屾暣 38 棰樻暟鎹紙鍚?mfg / S10锛夎 `01_STATUS.md` 涓?`handoff_scores.json`銆?
---

## 浜屻€佽繖涓寘鏈変粈涔堢敤

**鍒汉鎺ユ墜鏃舵渶闇€瑕佺殑涓変欢浜嬶紝杩欓噷閮芥湁锛?*

1. **鑳界洿鎺ユ彁浜ょ殑浜х墿** 鈥斺€?`*/zips/<op>_7files.zip` 灏辨槸鍒ら骞冲彴瑕佺殑 7 鏂囦欢鍖咃紙mfg 涓?4 鏂囦欢鍖咃級锛屽凡鏍￠獙銆屾潯鐩暟 / 璺緞鍓嶇紑 / 涓庢簮鐮?md5 涓€鑷?/ 鏃犺繚瑙勬爣璇嗙銆嶃€?2. **婧愮爜** 鈥斺€?`*/<op>/op_host|op_kernel/` 涓嬬殑鍙紪杈戞枃浠讹紝鍙洿鎺ユ敼銆?3. **韪╄繃鐨勫潙** 鈥斺€?`02_PITFALLS.md`锛堣蒋纭欢 + 鍒ら鏈鸿涓猴級鍜?`03_GOLDEN_ORDER.md`锛?*golden 褰掔害椤哄簭鐨勯€嗗悜缁撹锛岃繖鏄湰璧涙渶璐电殑涓滆タ**锛夈€?
---

## 涓夈€佺洰褰曠粨鏋?
```
.
鈹溾攢鈹€ README.md                 鈫?浣犲湪杩欓噷
鈹溾攢鈹€ 01_STATUS.md              鈫?38 棰樺畬鏁存垚缁╄〃锛堟帓鍚?寰楀垎/status/鍧?case 鏁帮級
鈹溾攢鈹€ 02_PITFALLS.md            鈫?纭欢 & 鍒ら鏈哄潙锛堝繀璇伙級
鈹溾攢鈹€ 03_GOLDEN_ORDER.md        鈫?golden 褰掔害椤哄簭閫嗗悜锛堝繀璇伙級
鈹溾攢鈹€ 04_TOOLS.md               鈫?宸ュ叿鑴氭湰璇存槑
鈹溾攢鈹€ 05_NEXT_STEPS.md          鈫?鏈繃 6 閬撻鐨勫崱鐐瑰垎鏋愪笌寤鸿
鈹溾攢鈹€ handoff_scores.json       鈫?骞冲彴鎶撳彇鐨勬垚缁╁師濮嬫暟鎹?鈹溾攢鈹€ _manifest.json            鈫?姣忎釜绠楀瓙鐨勬潵婧愮洰褰?/ 鏂囦欢娓呭崟 / 瀹屾暣鎬?鈹溾攢鈹€ ladder/                   鈫?澶╂璧?18 涓畻瀛?鈹?  鈹溾攢鈹€ <op>/op_host/*.cpp, op_kernel/*.h|*.cpp
鈹?  鈹斺攢鈹€ zips/<op>_7files.zip  鈫?鐩存帴鍙彁浜?鈹溾攢鈹€ mfg/                      鈫?鍒堕€犱笟璧?15 涓畻瀛?鈹?  鈹溾攢鈹€ <op>/...
鈹?  鈹斺攢鈹€ zips/<op>_7files.zip
鈹溾攢鈹€ s10/                      鈫?S10 浜斾釜绠楀瓙
鈹?  鈹溾攢鈹€ <op>/...
鈹?  鈹斺攢鈹€ zips/<op>_7files.zip
鈹溾攢鈹€ tools/                    鈫?鎻愪氦 / 鏍￠獙 / 璇婃柇鑴氭湰
鈹斺攢鈹€ docs/                     鈫?璧涢瑙ｆ瀽銆佷綔鎴樻墜鍐屻€佸悇棰?NOTES銆佹帓琛屾蹇収
```

---

## 鍥涖€? 鍒嗛挓涓婃墜

```bash
# 1) 鏍￠獙鎵€鏈?zip 鏄惁瀹屽ソ锛堟潯鐩暟銆乵d5銆佽繚瑙勬爣璇嗙锛?python tools/verify_handoff.py

# 2) 鎻愪氦鏌愰亾棰橈紙鑷姩澶勭悊 429 鍐峰嵈銆佽疆璇㈤€?case 缁撴灉锛?python tools/subpoll.py trace
#   鎴栦粠鎸囧畾 zip 鎻愪氦锛圓/B 鍙樹綋锛?python tools/sub_zip.py <zip璺緞> <problemId> <tag>

# 3) 鐪嬫煇娆℃彁浜ょ殑閫?case 璇︽儏锛堝惈鐪熸満 profiling msg锛?python tools/one_detail.py <submissionId>

# 4) 鎻愪氦鍓嶅繀璺戠殑涓変欢濂?python tools/scan_badwords.py    # 杩濊鏍囪瘑绗︼紙cout/cin/printf/getenv鈥︼級
python tools/verify_zips.py      # zip 涓庢簮鐮佷竴鑷?python tools/verify_handoff.py   # 鏈寘缁撴瀯鏍￠獙
```

**杩滅寮€鍙戞満**锛堝師鐜锛孋ANN 9.2 / 910B2锛夛細
`plink -ssh -P 40012 -pw <pw> root@<DEV_HOST>`锛屽伐绋嬪湪 `/root/work/ladder/<op>`銆乣/root/work/mfg/<op>`銆?> 娉ㄦ剰锛氫氦鎺ユ椂璇ユ満 SSH 绔彛锛?0012锛夊凡鎷掕繛锛岃嫢闇€閲嶈繛璇峰厛纭绔彛杞彂鏄惁鎭㈠銆?
---

## 浜斻€佸繀椤荤煡閬撶殑鍥涗欢浜嬶紙鍏堣杩欏嚑鏉★紝鑳界渷鍑犲ぉ锛?
1. **鍒ゆ嵁鏄€岀浉瀵硅宸?< 1e-4銆?* 涓?**銆岀粷瀵硅宸?< 1e-4銆?*锛堜笉鏄?numpy allclose 鐨?`atol+rtol路|b|` 鎴栫粍鍚堬級銆?   鈫?杩戦浂杈撳嚭鍏冪礌鍙宸?1 ulp 灏变細鎸?鈫?楂樼簿搴﹂噺鍖栫被绠楀瓙鍩烘湰鍙湁**閫愪綅澶嶅埢 golden** 鎵嶈兘杩囥€?
2. **銆屾彁楂樼簿搴︺€嶆槸閿欑殑锛屻€屽鍒?golden 鐨勫綊绾﹂『搴忋€嶆墠鏄鐨勩€?*
   瀹炴祴锛歠p64 鐪熷€肩畻鍑烘潵鐨勭粨鏋滄瘮绾崌搴?fp32 閿欏緱鏇村锛坄n4_T32768` 鐢ㄤ緥锛歠p64 閿?148 涓厓绱狅紝鍗囧簭閿?0 涓級銆?   鍥犱负 golden 鏈韩灏辨槸鏌愪釜鐗瑰畾鑸嶅叆椤哄簭鐨勪骇鐗┿€傝瑙?`03_GOLDEN_ORDER.md`銆?
3. **鎻愪氦鍓嶅繀鏌ヨ繚瑙勬爣璇嗙**锛歚cout` / `cin` / `cerr` / `printf` / `getenv` / `system` / `popen`
   琚垽銆屾彁浜や唬鐮佷腑瀛樺湪涓嶅悎瑙勫唴瀹广€嶁啋 **HTTP 400 涓旂収鏍锋墸閰嶉**銆傛妸 `cout`锛坈hannel-out锛夋敼鍚嶄负 `chOut` 鍗冲彲銆?
4. **鍒ら鏈烘湁 3 绉掔‖鐪嬮棬鐙?*锛坄aclrtSynchronizeStreamWithTimeout(stream, 3000)` 鈫?`ret=507046`锛夌敤浜?cube 绫荤畻瀛愶紱
   浣嗗叾瀹冪畻瀛愮殑 harness 瀹藉寰楀锛堝疄娴嬫湁 case 璺?272 绉掍粛 Pass锛夈€?*涓嶈鎶婃€ц兘浼樺寲鍑岄┚浜庢纭€т箣涓?*銆?
---

## 鍏€佷氦鎺ユ椂闇€瑕佸悜鐢ㄦ埛纭鐨勪簨

- CANNJudge 鐨?JWT / userId 鏄惁浠嶆湁鏁堬紙鏈寘 `tools/` 閲屽啓姝荤殑鏄?2026-11-25 鐨勫揩鐓э紝`exp` 鍒?2026-11-26 鍓嶅悗锛?*寰堝彲鑳藉凡杩囨湡锛岄渶瑕侀噸鏂扮櫥褰曞彇 Cookie**锛夈€?- 鍘熷紑鍙戞満 `<DEV_HOST>:40012` 鐨勭鍙ｈ浆鍙戞槸鍚﹂渶瑕侀噸鏂板紑閫氥€?- 澶╂璧涙槸 **`best` 妯″紡璁″垎**锛堝彇鍘嗗彶鏈€浼橈級锛屾墍浠ュ弽澶嶆彁浜?A/B 涓嶄細鎺夊垎锛沵fg / S10 璇疯嚜琛岀‘璁よ鍒嗗彛寰勩€?
---

## 涓冦€佸叕寮€浠撳簱璇存槑

鏈粨搴撲负**鍏紑**鍙戝竷锛屽洜姝ゅ凡鍋氫互涓嬪鐞嗭細

- `tools/` 涓?`docs/` 涓殑 **CANNJudge JWT銆佸紑鍙戞満 SSH 瀵嗙爜 / hostkey / 鍐呯綉 IP** 宸插叏閮ㄦ浛鎹负鍗犱綅绗?  锛坄<CANNJUDGE_JWT_REDACTED>` / `<SSH_PASSWORD_REDACTED>` 绛夛級锛?*杩愯鑴氭湰鍓嶈鎸?`tools/CREDENTIALS.md` 濉洖**銆?- 鎻愪氦浜х墿 zip锛坄*/zips/*.zip`锛変笌婧愮爜涓嶅惈浠讳綍鍑嵁锛屽彲鐩存帴浣跨敤銆?
---

## 鍏€佺洰褰曢噷鍚?zip 鏄粈涔?
`ladder/zips/<op>_7files.zip` 灏辨槸鍒ら骞冲彴銆屾墦鍖呬笂浼犲伐绋嬨€嶈鐨勪笢瑗匡細
7 涓潯鐩紝璺緞褰㈠ `op_host/<base>_def.cpp`銆乣op_kernel/<base>.h`锛?*鏃犲伐绋嬪悕鍓嶇紑**銆?
mfg 閫氬父鏄?4 涓潯鐩細`op_host/<base>.cpp` + `op_kernel/<base>.cpp` + `op_kernel/<base>_tiling.h` + `op_kernel/tiling_key_<base>.h`銆?
鑷锛歚python tools/verify_handoff.py`锛堟牎楠屾潯鐩暟 / md5 / 杩濊鏍囪瘑绗︼級銆?
---

## 涔濄€佸叧浜?4 涓?mfg 绠楀瓙鐨?3 鏂囦欢鍖?
`random_walk` / `cylinder_query` / `spmm_sum` / `segment_min_csr` 杩?4 閬?*鏈潵灏辨病鏈?* `tiling_key_<base>.h`
锛堝崟 tiling key锛屾棤闇€妯℃澘鍙傛暟锛夛紝鍥犳瀹冧滑鐨?zip 鏄?**3 涓潯鐩?* 鈥斺€?杩欐槸瀹冧滑褰撳勾鎻愪氦鏃剁殑鐪熷疄褰㈡€侊紝涓嶆槸缂哄け銆?鍏朵綑 mfg 绠楀瓙涓?4 涓潯鐩€?
