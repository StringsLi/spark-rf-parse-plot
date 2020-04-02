package com.strings.algo.tree.util;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.lang3.StringUtils;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class GsonUtil {

    private static Gson gson = new Gson();

    public static <T> Map<String, T> toMap(String json) {
        if (StringUtils.isEmpty(json)) {
            return Collections.emptyMap();
        }
        return gson.fromJson(json, new TypeToken<Map<String, T>>() {
        }.getType());
    }

    public static Map<String, String> toMapString(String json) {
        if (StringUtils.isEmpty(json)) {
            return Collections.emptyMap();
        }
        Map<String, Object> tmpMap = toMap(json);
        Map<String, String> map  = new HashMap<String, String>();
        Object[] keys = tmpMap.keySet().toArray();
        for (Object key: keys) {
            map.put(key.toString(), tmpMap.get(key).toString());
        }
        return map;
    }

}
