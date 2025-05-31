# 假设你的目录结构在 ~/rock/data/ 下
cd ~/rock/data/

echo "统计 Train 集合："
find train -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l

echo "统计 Valid 集合："
find valid -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l

echo "统计 Test 集合："
find test -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l

echo ""
echo "统计每一类样本数（Train + Valid + Test）:"
for d in train/* valid/* test/*; do
  echo "$d: $(find "$d" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)"
done
